"""Training data from the latent shards: dialect-tagged text + AudioVAE latents, voice cloning from the speaker's
other clips.

Every corpus's latents are held in memory (fp16, ~3.2 KB per second of audio: ~7 GB for 640 h). A sample is the
upstream layout (pack.py): with probability p_ref the reference layout with a 3-10 s window of another clip of the
same speaker, otherwise the plain one. The text starts with the dialect's tag token, dropped with probability p_no_tag
so that the model also speaks untagged text.

Corpus weights (``row_weights``): a row of weight w is drawn w times per epoch in expectation (floor(w) copies, one more
with probability frac(w), redrawn each epoch), so up- and down-weighting needs no copies on disk. Reference
augmentation (``p_ref_aug``, augment.py): the reference of a ``ref`` sample is degraded with probability p_ref_aug,
either marked for the training step (``ref_aug`` track) or taken from precomputed degraded variants.
"""

from __future__ import annotations

import glob
import json
import os
import random
import re
from collections import defaultdict

import torch

from arabic_voxcpm.data import manifest
from arabic_voxcpm.eval.metrics import normalize_ar
from arabic_voxcpm.train import dialects, pack

FPS = 25  # latent frames per second
_DIGIT = re.compile(r"[0-9\u0660-\u0669\u06f0-\u06f9]")


def select_rows(corpora=manifest.CORPORA, heldout: str | None = None, min_s: float = 1.0, max_s: float = 20.0,
                quality: str | None = None, max_cer: float = 0.10, min_dnsmos: float = 3.0) -> list[dict]:
    """Training rows: a known dialect, 1-20 s, plausible speaking rate, no digits, nothing of the evaluation set.

    quality: data/quality.py's scores; a real clip is kept only if it was scored (the scored candidates are already
    capped per speaker and dialect) with Cohere CER <= max_cer and DNSMOS >= min_dnsmos. Synthetic clips were
    filtered when made (synth/filter.py)."""
    held = json.load(open(heldout, encoding="utf-8")) if heldout else {"speakers": [], "uids": [], "texts": []}
    spk, uids, texts = set(held["speakers"]), set(held["uids"]), set(held["texts"])
    good = None
    if quality:
        good = set()
        for line in open(quality, encoding="utf-8"):
            q = json.loads(line)
            if q["cer"] <= max_cer and q["dnsmos_ovrl"] >= min_dnsmos:
                good.add(q["uid"])
    out = []
    for c in corpora:
        for r in manifest.rows(c):
            d = dialects.of_row(r)
            sec = r["frames"] / FPS
            if (d is None or r.get("split") == "test" or r["speaker"] in spk or r["uid"] in uids
                    or not (min_s <= sec <= max_s) or not r["text"]):
                continue
            if good is not None and c != "synth" and r["uid"] not in good:
                continue
            if not 2.0 <= len(r["text"]) / sec <= 35.0 or normalize_ar(r["text"]) in texts or _DIGIT.search(r["text"]):
                continue  # digits: the audio reads them in dialect, requests get them spelled out (text normalization)
            out.append({"uid": r["uid"], "source": c, "dialect": d, "speaker": f"{c}:{r['speaker']}",
                        "text": r["text"], "frames": r["frames"], "split": r.get("split", "train")})
    return out


def parse_weights(spec: str | None) -> dict[str, float]:
    """"synth=2,sada=0.5,synth:hejazi=3" -> {"synth": 2.0, "sada": 0.5, "synth:hejazi": 3.0} (corpus or
    corpus:dialect; the more specific one wins, unnamed rows weigh 1)."""
    out = {}
    for part in filter(None, (spec or "").replace(" ", "").split(",")):
        k, v = part.split("=")
        c, _, d = k.partition(":")
        if c not in manifest.CORPORA + ("synth",) or (d and dialects.ALIASES.get(d, d) not in dialects.DIALECTS):
            raise ValueError(f"unknown corpus or dialect in weight {part!r}")
        if float(v) < 0:
            raise ValueError(f"negative weight {part!r}")
        out[f"{c}:{dialects.ALIASES.get(d, d)}" if d else c] = float(v)
    return out


def row_weights(rows: list[dict], weights: dict[str, float]) -> list[float]:
    return [weights.get(f"{r['source']}:{r['dialect']}", weights.get(r["source"], 1.0)) for r in rows]


def hours(rows: list[dict], weights: list[float] | None = None) -> dict[str, float]:
    """(Expected, if weighted) hours per epoch per corpus and per corpus:dialect."""
    out = defaultdict(float)
    for i, r in enumerate(rows):
        h = r["frames"] / FPS / 3600 * (1.0 if weights is None else weights[i])
        out[r["source"]] += h
        out[f"{r['source']}:{r['dialect']}"] += h
    return {k: round(v, 2) for k, v in sorted(out.items())}


class LatentStore:
    def __init__(self, corpora=manifest.CORPORA, root: str = manifest.ROOT):
        self.parts: list[torch.Tensor] = []
        self.index: dict[str, tuple[int, int, int]] = {}
        for c in corpora:
            for path in sorted(glob.glob(os.path.join(root, c, "latents", "*.pt"))):
                d = torch.load(path, map_location="cpu", weights_only=True)
                k = len(self.parts)
                self.parts.append(d["latents"])
                off = d["offsets"].tolist()
                for i, u in enumerate(d["uids"]):
                    self.index[u] = (k, off[i], off[i + 1])

    def __contains__(self, uid: str) -> bool:
        return uid in self.index

    def __getitem__(self, uid: str) -> torch.Tensor:
        k, a, b = self.index[uid]
        return self.parts[k][a:b]


class TrainSet(torch.utils.data.Dataset):
    """Index i + k * len (k >= 1) is the k-th extra copy of row i in an epoch (weighted sampling): its own draws."""

    def __init__(self, rows: list[dict], store: LatentStore, tokenize, p_ref: float = 0.7, p_no_tag: float = 0.1,
                 ref_s: tuple[float, float] = (3.0, 10.0), seed: int = 0, p_ref_aug: float = 0.0,
                 aug_variants: list[LatentStore] | None = None):
        self.rows = [r for r in rows if r["uid"] in store]
        self.store, self.p_ref, self.p_no_tag = store, p_ref, p_no_tag
        # reference augmentation: precomputed degraded variants (augment.py), else marked for the training step
        self.p_ref_aug, self.aug_variants = p_ref_aug, aug_variants or []
        self.ref_frames = (int(ref_s[0] * FPS), int(ref_s[1] * FPS))
        self.seed, self.epoch = seed, 0
        # as inference tokenizes them: tokenize(tag + text), and the text alone for the untagged samples
        self.tagged_ids = [tokenize(dialects.tag_token(r["dialect"]) + r["text"]) for r in self.rows]
        self.text_ids = [tokenize(r["text"]) for r in self.rows]
        by = defaultdict(list)
        for i, r in enumerate(self.rows):
            by[r["speaker"]].append(i)
        self.same_speaker = [by[r["speaker"]] for r in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def length(self, i: int) -> int:
        """Upper bound of the packed length (with the longest reference)."""
        return len(self.tagged_ids[i]) + 7 + (self.rows[i]["frames"] + 3) // 4 + (self.ref_frames[1] + 3) // 4

    def __getitem__(self, idx: int) -> dict:
        i, copy = idx % len(self.rows), idx // len(self.rows)
        rng = random.Random((self.seed, self.epoch, i).__hash__() if copy == 0 else (self.seed, self.epoch, i, copy).__hash__())
        r = self.rows[i]
        ids = self.tagged_ids[i] if rng.random() >= self.p_no_tag else self.text_ids[i]
        target = pack.patches(self.store[r["uid"]].float())
        others = [j for j in self.same_speaker[i] if j != i]
        marks = self.p_ref_aug > 0 and not self.aug_variants  # the training step degrades marked references
        if others and rng.random() < self.p_ref:
            ref_uid = self.rows[rng.choice(others)]["uid"]
            ref = self.store[ref_uid].float()
            n = min(len(ref), rng.randint(*self.ref_frames))
            a = rng.randint(0, len(ref) - n)
            aug = self.p_ref_aug > 0 and rng.random() < self.p_ref_aug
            if aug and self.aug_variants:
                v = self.aug_variants[rng.randrange(len(self.aug_variants))]
                if ref_uid in v:  # same window of the degraded clip
                    ref = v[ref_uid].float()
            seq = pack.with_ref(pack.patches(ref[a : a + n]), ids, target)
            if marks:
                # at each reference patch its real frames (the window's last patch may be zero-padded), 0 elsewhere
                seq["ref_aug"] = torch.zeros(len(seq["text_tokens"]), dtype=torch.int32)
                if aug:
                    k = (n + pack.PATCH - 1) // pack.PATCH
                    seq["ref_aug"][1 : 1 + k] = pack.PATCH
                    seq["ref_aug"][k] = n - pack.PATCH * (k - 1)
            return seq
        seq = pack.plain(ids, target)
        if marks:
            seq["ref_aug"] = torch.zeros(len(seq["text_tokens"]), dtype=torch.int32)
        return seq


class TokenBudgetBatches(torch.utils.data.Sampler):
    """Batches of similar length whose padded size (batch x longest) stays within ``budget`` positions."""

    def __init__(self, ds: TrainSet, budget: int, seed: int = 0, pool: int = 2048, weights: list[float] | None = None):
        """weights: per row, the expected number of draws per epoch (None: each row once, the unweighted sampler)."""
        self.ds, self.budget, self.seed, self.pool, self.epoch = ds, budget, seed, pool, 0
        self.lengths = [ds.length(i) for i in range(len(ds))]
        assert weights is None or len(weights) == len(ds)
        self.weights = weights

    def indices(self, rng: random.Random) -> list[int]:
        """The epoch's draws: row i, and i + k * len for its k-th extra copy."""
        if self.weights is None:
            return list(range(len(self.ds)))
        n, out = len(self.ds), []
        for i, w in enumerate(self.weights):
            copies = int(w) + (rng.random() < w - int(w))
            out.extend(i + k * n for k in range(copies))
        return out

    def batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        idx = self.indices(rng)
        rng.shuffle(idx)
        out = []
        n = len(self.ds)
        for k in range(0, len(idx), self.pool):
            chunk = sorted(idx[k : k + self.pool], key=lambda i: self.lengths[i % n])
            batch, longest = [], 0
            for i in chunk:
                longest_new = max(longest, self.lengths[i % n])
                if batch and longest_new * (len(batch) + 1) > self.budget:
                    out.append(batch)
                    batch, longest_new = [], self.lengths[i % n]
                batch.append(i)
                longest = longest_new
            if batch:
                out.append(batch)
        rng.shuffle(out)
        return out

    def __iter__(self):
        return iter(self.batches())

    def __len__(self) -> int:
        return len(self.batches())

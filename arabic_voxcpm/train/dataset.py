"""Training data from the latent shards: dialect-tagged text + AudioVAE latents, voice cloning from the speaker's
other clips.

Every corpus's latents are held in memory (fp16, ~3.2 KB per second of audio: ~7 GB for 640 h). A sample is the
upstream layout (pack.py): with probability p_ref the reference layout with a 3-10 s window of another clip of the
same speaker, otherwise the plain one. The text starts with the dialect's tag token, dropped with probability p_no_tag
so that the model also speaks untagged text.
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
    def __init__(self, rows: list[dict], store: LatentStore, tokenize, p_ref: float = 0.7, p_no_tag: float = 0.1,
                 ref_s: tuple[float, float] = (3.0, 10.0), seed: int = 0):
        self.rows = [r for r in rows if r["uid"] in store]
        self.store, self.p_ref, self.p_no_tag = store, p_ref, p_no_tag
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

    def __getitem__(self, i: int) -> dict:
        rng = random.Random((self.seed, self.epoch, i).__hash__())
        r = self.rows[i]
        ids = self.tagged_ids[i] if rng.random() >= self.p_no_tag else self.text_ids[i]
        target = pack.patches(self.store[r["uid"]].float())
        others = [j for j in self.same_speaker[i] if j != i]
        if others and rng.random() < self.p_ref:
            ref = self.store[self.rows[rng.choice(others)]["uid"]].float()
            n = min(len(ref), rng.randint(*self.ref_frames))
            a = rng.randint(0, len(ref) - n)
            return pack.with_ref(pack.patches(ref[a : a + n]), ids, target)
        return pack.plain(ids, target)


class TokenBudgetBatches(torch.utils.data.Sampler):
    """Batches of similar length whose padded size (batch x longest) stays within ``budget`` positions."""

    def __init__(self, ds: TrainSet, budget: int, seed: int = 0, pool: int = 2048):
        self.ds, self.budget, self.seed, self.pool, self.epoch = ds, budget, seed, pool, 0
        self.lengths = [ds.length(i) for i in range(len(ds))]

    def batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        idx = list(range(len(self.ds)))
        rng.shuffle(idx)
        out = []
        for k in range(0, len(idx), self.pool):
            chunk = sorted(idx[k : k + self.pool], key=lambda i: self.lengths[i])
            batch, longest = [], 0
            for i in chunk:
                longest_new = max(longest, self.lengths[i])
                if batch and longest_new * (len(batch) + 1) > self.budget:
                    out.append(batch)
                    batch, longest_new = [], self.lengths[i]
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

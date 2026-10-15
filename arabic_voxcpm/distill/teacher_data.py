"""Teacher data for the LocDiT students (issues #6, #12): the teacher's own sampler output for real contexts.

    python -m arabic_voxcpm.distill.teacher_data --teacher adapters --model /workspace/models/VoxCPM2-Arabic-Dialects \\
        --out /workspace/data/arabic/teacher/adapters --patches 1500000 --verify 2000
    python -m arabic_voxcpm.distill.teacher_data --teacher <merged tagged model> --out ...      (a tagged teacher)

Contexts: the teacher-v2 training selection (train.train's --corpora / --quality / --max-cer / --min-dnsmos, the
evaluation set's heldout.json out), packed as in training: 70 % clone another clip of the same speaker by reference
(the ``ref`` layout), the rest plain. Clips are drawn with replacement, evenly over dialects by default (--balance).
- ``--teacher adapters``: the base model with the current dialect adapters. Each clip goes through the adapter of
  its dialect (train.adapters.ADAPTER_OF_DIALECT, from eval.build.CURRENT_ADAPTER; --adapter-map overrides), dialects
  without one through the base model; the text has no tag. Batches hold one adapter each.
- a merged directory: a tagged teacher (train.model.export_merged), the dialect's tag in front of the text.
The teacher's LMs run teacher-forced over each batch (with the batch's adapter) and give, at every target patch, the
LocDiT's conditioning mu and the previous real patch (cond). From seeded noise z the teacher's sampler (sampler.py:
10 Euler steps, CFG 2.0, CFG-zero*, as inference) solves the patch x, with the adapter's LocDiT LoRA active.

Written to --out: clips.jsonl (every selected clip: id, uid, text, dialect, adapter, speaker), shard-NNNNN.pt of
50k rows each (the last one shorter), manifest.json (configuration, plan fingerprint, shards, rows per adapter,
the position to resume from). Per row: mu [C], cond / z / x (the teacher's patch) / target (the real patch) [D, P],
all bf16 (z is rounded to bf16 before solving, so the recorded z reproduces x); adapter (index into
manifest["adapters"], -1 = base model), clip and ref (ids in clips.jsonl; ref = the cloned clip, -1 = no cloning),
ref_start / ref_frames (the reference window, latent frames), rep (which draw of the clip), patch (index in the
clip), batch / row (position in the plan), seed (the batch's noise seed). The disk use is printed (and checked
against the free space) before generating. An interrupted run resumes after its last complete shard (same
configuration: the plan is deterministic, its fingerprint is checked).

--verify N: replays recorded batches (>= N rows) through the teacher as they were made (their adapter, z, mu, cond, the
same sampler chunks) and asserts that every solved patch reproduces the recorded x within --verify-tol (relative RMS
per row; x is stored in bf16), and that the same rows replayed without their adapter's LoRA are further off (the
LoRA was active when they were made). (Re-solved in other batch shapes, bf16 kernels differ in rounding and a few
percent of the patches, where the guided 10-step solve is sensitive, move by more.)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import time
from collections import Counter, defaultdict

import torch

from arabic_voxcpm.distill import sampler

SHARD_ROWS = 50_000
VECTORS = ("mu", "cond", "z", "x", "target")  # bf16
META = {"adapter": torch.int16, "clip": torch.int32, "ref": torch.int32, "ref_start": torch.int16,
        "ref_frames": torch.int16, "rep": torch.int16, "patch": torch.int16, "batch": torch.int32,
        "row": torch.int32, "seed": torch.int64}
DEFAULT_MODEL = "/workspace/models/VoxCPM2-Arabic-Dialects"


def bytes_per_row(mu_dim: int = 2048, feat_dim: int = 64, patch: int = 4) -> int:
    return 2 * mu_dim + 4 * 2 * feat_dim * patch + sum(torch.empty(0, dtype=d).element_size() for d in META.values())


class PlainTokenizer:
    """Text -> ids as the inference engine tokenizes for a model without tags (engine.py: the upstream wrapper over
    LlamaTokenizerFast.from_pretrained(model))."""

    def __init__(self, model_dir: str):
        from transformers import LlamaTokenizerFast
        from voxcpm.model.utils import mask_multichar_chinese_tokens

        self.wrapped = mask_multichar_chinese_tokens(LlamaTokenizerFast.from_pretrained(model_dir))

    def __call__(self, text: str) -> list[int]:
        return list(self.wrapped(text))


def weights(rows: list[dict], adapters: list[str | None], balance: str) -> list[float]:
    """Draw weights: dialect = every dialect equally likely; adapter = every adapter (and the base model) equally
    likely, its dialects evenly; none = every clip equally likely."""
    if balance == "none":
        return [1.0] * len(rows)
    count = Counter(r["dialect"] for r in rows)
    if balance == "dialect":
        return [1.0 / count[r["dialect"]] for r in rows]
    per = defaultdict(set)
    for r, a in zip(rows, adapters):
        per[a].add(r["dialect"])
    return [1.0 / (len(per[a]) * count[r["dialect"]]) for r, a in zip(rows, adapters)]


def pack_batches(lengths: list[int], budget: int, rng: random.Random, pool: int = 2048) -> list[list[int]]:
    """Batches (indices into lengths) of similar length whose padded size stays within budget (TokenBudgetBatches)."""
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    out = []
    for k in range(0, len(idx), pool):
        batch, longest = [], 0
        for i in sorted(idx[k : k + pool], key=lambda i: lengths[i]):
            new = max(longest, lengths[i])
            if batch and new * (len(batch) + 1) > budget:
                out.append(batch)
                batch, new = [], lengths[i]
            batch.append(i)
            longest = new
        if batch:
            out.append(batch)
    return out


def make_plan(rows: list[dict], adapters: list[str | None], datasets: dict, n_draws: int, budget: int,
              balance: str, seed: int) -> list[tuple[str | None, list[tuple[int, int]]]]:
    """[(adapter, [(clip id, rep), ...]), ...]: n_draws clips drawn with replacement (``weights``), packed per adapter
    into token-budget batches, the batches shuffled. datasets[adapter] = (TrainSet, its clip ids)."""
    rng = random.Random(seed)
    drawn = rng.choices(range(len(rows)), weights=weights(rows, adapters, balance), k=n_draws)
    reps = Counter()
    by = defaultdict(list)
    for g in sorted(drawn):
        by[adapters[g]].append((g, reps[g]))
        reps[g] += 1
    plan = []
    for a in sorted(by, key=lambda a: (a is not None, a or "")):
        ds, ids = datasets[a]
        local = {g: i for i, g in enumerate(ids)}
        samples = by[a]
        lengths = [ds.length(local[g]) for g, _ in samples]
        for b in pack_batches(lengths, budget, random.Random(f"{seed}-{a}")):
            plan.append((a, [samples[i] for i in b]))
    rng.shuffle(plan)
    return plan


def fingerprint(plan, extra: dict) -> str:
    h = hashlib.sha1(json.dumps(extra, sort_keys=True).encode())
    for a, b in plan:
        h.update(f"{a}:{b}".encode())
    return h.hexdigest()


class Writer:
    """Shards of SHARD_ROWS rows and manifest.json, written atomically shard by shard; resumable."""

    def __init__(self, out: str, meta: dict, shard_rows: int = SHARD_ROWS):
        self.out, self.shard_rows = out, shard_rows
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, "manifest.json")
        if os.path.exists(path):
            m = json.load(open(path))
            if m["fingerprint"] != meta["fingerprint"]:
                raise SystemExit(f"{out} holds teacher data of another configuration (plan {m['fingerprint'][:12]}, "
                                 f"asked {meta['fingerprint'][:12]}): another --out, or remove it")
            self.m = m
        else:
            self.m = dict(meta, shards=[], rows=0, rows_by_adapter={}, next=[0, 0], complete=False)
        self.buf: list[dict] = []

    @property
    def rows(self) -> int:
        return self.m["rows"] + sum(len(b["mu"]) for b in self.buf)

    @property
    def resume(self) -> tuple[int, int]:
        """(batch, row) of the first row not in a written shard."""
        return tuple(self.m["next"])

    def add(self, rows: dict) -> None:
        if len(rows["mu"]):
            self.buf.append(rows)
        while sum(len(b["mu"]) for b in self.buf) >= self.shard_rows:
            self._write(self.shard_rows)

    def close(self) -> None:
        """The rest as a last, shorter shard; the run is complete."""
        if self.buf:
            self._write(sum(len(b["mu"]) for b in self.buf))
        self.m["complete"] = True
        self._save_manifest()

    def _write(self, n: int) -> None:
        cat = {k: torch.cat([b[k] for b in self.buf]) for k in self.buf[0]}
        shard = {k: v[:n].clone() for k, v in cat.items()}
        rest = {k: v[n:] for k, v in cat.items()}
        self.buf = [rest] if len(rest["mu"]) else []
        name = f"shard-{len(self.m['shards']):05d}.pt"
        torch.save(shard, os.path.join(self.out, name + ".tmp"))
        os.replace(os.path.join(self.out, name + ".tmp"), os.path.join(self.out, name))
        for a, c in Counter(shard["adapter"].tolist()).items():
            self.m["rows_by_adapter"][str(a)] = self.m["rows_by_adapter"].get(str(a), 0) + c
        nxt = [int(shard["batch"][-1]), int(shard["row"][-1]) + 1]
        self.m["shards"].append({"file": name, "rows": n, "next": nxt})
        self.m["rows"] += n
        self.m["next"] = nxt
        self._save_manifest()

    def _save_manifest(self) -> None:
        path = os.path.join(self.out, "manifest.json")
        json.dump(self.m, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)


def solve_chunked(est, z, mu, cond, cfg: float, steps: int, chunk: int) -> torch.Tensor:
    return torch.cat([sampler.solve(est, z[a : a + chunk], mu[a : a + chunk], cond[a : a + chunk], cfg_value=cfg,
                                    steps=steps) for a in range(0, len(z), chunk)])


def batch_seed(seed: int, bi: int) -> int:
    return seed * 1_000_003 + bi


def batch_rows(lm_mask: torch.Tensor, samples, infos, ids, bi: int, adapter_idx: int, seed: int) -> dict:
    """The meta columns of a batch's rows (row-major over its loss positions, the order of dit_inputs' mu).
    samples: [(clip id, rep)]; infos: TrainSet.sample's, ref as an index into ids (the dataset's clip ids)."""
    first: dict[int, int] = {}
    cols = {k: [] for k in ("clip", "ref", "ref_start", "ref_frames", "rep", "patch")}
    pos = lm_mask.nonzero().tolist()
    for b, t in pos:
        first.setdefault(b, t)
        g, rep = samples[b]
        info = infos[b]
        cols["clip"].append(g)
        cols["ref"].append(-1 if info["ref"] is None else ids[info["ref"]])
        cols["ref_start"].append(info["ref_start"])
        cols["ref_frames"].append(info["ref_frames"])
        cols["rep"].append(rep)
        cols["patch"].append(t - first[b])
    n = len(pos)
    out = {k: torch.tensor(v, dtype=META[k]) for k, v in cols.items()}
    out["adapter"] = torch.full((n,), adapter_idx, dtype=META["adapter"])
    out["batch"] = torch.full((n,), bi, dtype=META["batch"])
    out["row"] = torch.arange(n, dtype=META["row"])
    out["seed"] = torch.full((n,), batch_seed(seed, bi), dtype=META["seed"])
    return out


def generate(plan, writer: Writer, run_batch, patches: int, log_every: int = 20) -> None:
    """run_batch(bi, adapter, samples) -> rows (dict of CPU tensors, row-major); resumes at writer.resume."""
    rb, rr = writer.resume
    t0, made = time.time(), 0
    for bi, (adapter, samples) in enumerate(plan):
        if writer.rows >= patches:
            break
        if bi < rb:
            continue
        rows = run_batch(bi, adapter, samples)
        if bi == rb and rr:
            keep = rows["row"] >= rr
            rows = {k: v[keep] for k, v in rows.items()}
        writer.add(rows)
        made += len(rows["mu"])
        if bi % log_every == 0:
            print(f"batch {bi}/{len(plan)}: {writer.rows} rows, {made / (time.time() - t0):.0f} rows/s", flush=True)
    writer.close()
    print(f"done: {writer.rows} rows in {len(writer.m['shards'])} shards, {made / (time.time() - t0):.0f} rows/s "
          f"-> {writer.out}", flush=True)


def load_batches(out: str, n: int, seed: int = 0) -> list[dict]:
    """Whole recorded batches (all fields), drawn at random until they hold >= n rows. A batch whose rows were
    resumed part-way (not rows 0..k-1) is passed over."""
    m = json.load(open(os.path.join(out, "manifest.json")))
    shards = [torch.load(os.path.join(out, s["file"]), map_location="cpu", weights_only=True, mmap=True)
              for s in m["shards"]]
    where = defaultdict(list)  # batch -> shards holding its rows
    for k, d in enumerate(shards):
        for b in d["batch"].unique().tolist():
            where[b].append(k)
    order = sorted(where)
    random.Random(seed).shuffle(order)
    picked, rows = [], 0
    for b in order:
        if rows >= n:
            break
        parts = [{k: v[d["batch"] == b] for k, v in d.items()} for d in (shards[i] for i in where[b])]
        d = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
        if not torch.equal(d["row"], torch.arange(len(d["row"]), dtype=d["row"].dtype)):
            continue
        picked.append(d)
        rows += len(d["row"])
    return picked


def rel_rms(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per row: RMS(a - b) / RMS(b)."""
    return ((a - b) ** 2).mean(dim=(1, 2)).sqrt() / (b**2).mean(dim=(1, 2)).sqrt().clamp(min=1e-6)


def replay(out: str, n: int, use, solve, device, tol: float = 0.02, seed: int = 0) -> dict:
    """The replay check: recorded batches (>= n rows) re-solved as they were made, with their adapter (use(name or
    None) switches the teacher; solve(z, mu, cond) -> x in the generator's chunks: the same shapes, so the same
    kernels), each row compared to its recorded x (relative RMS; the recorded x is bf16). The batches that have an
    adapter are also solved without it, which must be further off."""
    names = json.load(open(os.path.join(out, "manifest.json")))["adapters"]
    errs, errs_a, errs_base = [], [], []
    for d in load_batches(out, n, seed):
        a = int(d["adapter"][0])
        z, mu, cond, x = (d[k].to(device).float() for k in ("z", "mu", "cond", "x"))
        use(names[a] if a >= 0 else None)
        e = rel_rms(solve(z, mu, cond).float(), x).cpu()
        errs.append(e)
        if a >= 0:
            errs_a.append(e)
            use(None)
            errs_base.append(rel_rms(solve(z, mu, cond).float(), x).cpu())
    err = torch.cat(errs)
    res = {"rows": len(err), "batches": len(errs), "median": float(err.median()), "p99": float(err.quantile(0.99)),
           "max": float(err.max()), "tol": tol}
    if errs_base:
        res["with_adapter_median"] = float(torch.cat(errs_a).median())
        res["without_adapter_median"] = float(torch.cat(errs_base).median())
    print("replay: " + json.dumps(res), flush=True)
    assert res["max"] <= tol, f"replayed patches differ from the recorded ones: {res}"
    if errs_base:
        assert res["without_adapter_median"] > 3 * res["with_adapter_median"], \
            f"the adapter's LoRA does not explain the recorded patches: {res}"
    return res


@torch.no_grad()
def main(args) -> None:
    from arabic_voxcpm.train import adapters as A
    from arabic_voxcpm.train import dataset as D
    from arabic_voxcpm.train import dialects, pack
    from arabic_voxcpm.train.forward import dit_inputs

    dev = torch.device(args.device)
    adapter_mode = args.teacher == "adapters"
    model_dir = args.model if adapter_mode else args.teacher
    corpora = tuple(args.corpora.split(","))
    rows = D.select_rows(corpora, heldout=None if args.heldout == "none" else args.heldout, quality=args.quality,
                         max_cer=args.max_cer, min_dnsmos=args.min_dnsmos)
    store = D.LatentStore(corpora)
    n_sel = len(rows)
    rows = sorted((r for r in rows if r["uid"] in store), key=lambda r: (r["source"], r["uid"]))
    mapping = dict(A.ADAPTER_OF_DIALECT)
    if args.adapter_map:
        mapping.update(json.load(open(args.adapter_map)))
    available = A.find(model_dir) if adapter_mode else {}
    adapters = [A.adapter_of(r["dialect"], mapping) if adapter_mode else None for r in rows]
    unknown = sorted({a for a in adapters if a is not None} - set(available))
    if unknown:
        raise SystemExit(f"adapters {unknown} are not in {model_dir}/adapters ({sorted(available)})")
    names = sorted(available)
    hours = sum(r["frames"] for r in rows) / D.FPS / 3600
    print(f"{len(rows)} clips ({hours:.1f} h; {n_sel - len(rows)} selected clips without latents), "
          f"by adapter: {dict(Counter(a or 'base' for a in adapters))}", flush=True)

    if adapter_mode:
        tok = PlainTokenizer(model_dir)
    else:
        from arabic_voxcpm.train.model import Tokenizer

        tok = Tokenizer(model_dir)
    datasets = {}
    for a in sorted(set(adapters), key=lambda a: (a is not None, a or "")):
        ids = [g for g, x in enumerate(adapters) if x == a]
        ds = D.TrainSet([rows[g] for g in ids], store, tok, p_ref=args.p_ref, p_no_tag=0.0, seed=args.seed,
                        tagged=not adapter_mode)
        assert len(ds) == len(ids)
        datasets[a] = (ds, ids)
    local = {a: {g: i for i, g in enumerate(ids)} for a, (_, ids) in datasets.items()}  # clip id -> dataset index
    avg_patches = sum(r["frames"] for r in rows) / len(rows) / pack.PATCH
    plan = make_plan(rows, adapters, datasets, int(args.patches / avg_patches * 1.05) + 1, args.budget,
                     args.balance, args.seed)
    config = {k: getattr(args, k) for k in ("teacher", "corpora", "quality", "max_cer", "min_dnsmos", "heldout",
                                            "patches", "steps", "cfg", "p_ref", "budget", "dit_chunk", "balance",
                                            "seed")}
    config["model"] = model_dir
    config["adapter_map"] = {d: mapping.get(d) for d in dialects.DIALECTS} if adapter_mode else None
    meta = {"fingerprint": fingerprint(plan, config), "config": config, "adapters": names,
            "dialects": dialects.DIALECTS, "batches": len(plan), "clips": len(rows), "bytes_per_row": bytes_per_row(),
            "layout": {k: "bfloat16" for k in VECTORS} | {k: str(v).split(".")[-1] for k, v in META.items()}}
    writer = Writer(args.out, meta)
    clips = os.path.join(args.out, "clips.jsonl")
    if not os.path.exists(clips) or not writer.m["shards"]:
        with open(clips + ".tmp", "w", encoding="utf-8") as f:
            for g, (r, a) in enumerate(zip(rows, adapters)):
                f.write(json.dumps({"id": g, "uid": r["uid"], "source": r["source"], "dialect": r["dialect"],
                                    "adapter": a, "speaker": r["speaker"], "text": r["text"], "frames": r["frames"]},
                                   ensure_ascii=False) + "\n")
        os.replace(clips + ".tmp", clips)

    todo = 0 if writer.m["complete"] else max(0, args.patches - writer.m["rows"])
    need, free = todo * meta["bytes_per_row"], shutil.disk_usage(args.out).free
    print(f"disk: {args.patches} rows x {meta['bytes_per_row']} B = {args.patches * meta['bytes_per_row'] / 1e9:.2f} GB "
          f"in {-(-args.patches // SHARD_ROWS)} shards; {writer.m['rows']} rows written, {need / 1e9:.2f} GB to "
          f"write, {free / 1e9:.0f} GB free; plan: {len(plan)} batches", flush=True)
    if need > 0.95 * free:
        raise SystemExit("not enough disk space for the teacher data")
    if args.estimate_only:
        return

    if adapter_mode:
        model, bank = A.load_teacher(model_dir, dev, available)
        use = lambda name: bank.use(model, name)  # noqa: E731
    else:
        from voxcpm.model.voxcpm2 import VoxCPM2Model

        model = VoxCPM2Model.from_local(model_dir, optimize=False, device=str(dev)).eval()
        model.audio_vae = None
        use = lambda name: None  # noqa: E731
    est = model.feat_decoder.estimator
    index = {a: i for i, a in enumerate(names)}

    def run_batch(bi, adapter, samples):
        ds, ids = datasets[adapter]
        made = [ds.sample(local[adapter][g], rep) for g, rep in samples]
        batch = {k: v.to(dev, non_blocking=True) for k, v in pack.collate([s for s, _ in made]).items()}
        use(adapter)
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            mu, target, cond, _, lm_mask = dit_inputs(model, **batch)
        g = torch.Generator(device=dev).manual_seed(batch_seed(args.seed, bi))
        z = torch.randn(target.shape, device=dev, generator=g).to(torch.bfloat16)
        x = solve_chunked(est, z.float(), mu, cond.float(), args.cfg, args.steps, args.dit_chunk)
        out = batch_rows(lm_mask, samples, [i for _, i in made], ids, bi, index.get(adapter, -1), args.seed)
        out.update(mu=mu.to(torch.bfloat16).cpu(), cond=cond.to(torch.bfloat16).cpu(), z=z.cpu(),
                   x=x.to(torch.bfloat16).cpu(), target=target.to(torch.bfloat16).cpu())
        return out

    if not writer.m["complete"]:
        torch.manual_seed(args.seed)
        generate(plan, writer, run_batch, args.patches)
        if dev.type == "cuda":
            print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB allocated, "
                  f"{torch.cuda.max_memory_reserved() / 2**30:.2f} GiB reserved", flush=True)
    if args.verify:
        res = replay(args.out, args.verify, use,
                     lambda z, mu, cond: solve_chunked(est, z, mu, cond, args.cfg, args.steps, args.dit_chunk),
                     dev, args.verify_tol, args.seed)
        json.dump(res, open(os.path.join(args.out, "verify.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--teacher", required=True, help="'adapters' (--model's base + dialect adapters) or a merged "
                                                     "tagged model directory")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="--teacher adapters: the model with adapters/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--heldout", default="/workspace/data/arabic/eval/v2/heldout.json", help="'none' for smoke tests")
    ap.add_argument("--corpora", default="lahgtna,sada,synth", help="as train.train (teacher-v2: lahgtna,sada,synth)")
    ap.add_argument("--quality", default="/workspace/data/arabic/quality/scores.jsonl",
                    help="as train.train: real clips kept only if scored and passing ('' = no quality filter)")
    ap.add_argument("--max-cer", type=float, default=0.10)
    ap.add_argument("--min-dnsmos", type=float, default=3.0)
    ap.add_argument("--adapter-map", help="JSON {dialect: adapter or null} over the default routing")
    ap.add_argument("--balance", choices=["dialect", "adapter", "none"], default="dialect")
    ap.add_argument("--patches", type=int, default=1_500_000, help="rows (target patches) to record")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--p-ref", type=float, default=0.7)
    ap.add_argument("--budget", type=int, default=8000, help="padded positions per teacher-forced batch")
    ap.add_argument("--dit-chunk", type=int, default=1024, help="patches per sampler call (memory: ~0.4 GB per 512)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify", type=int, default=0, help="replay this many recorded rows through the teacher")
    ap.add_argument("--verify-tol", type=float, default=0.02, help="max relative RMS error of a replayed patch")
    ap.add_argument("--estimate-only", action="store_true", help="print the plan and the disk use, generate nothing")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    a.quality = a.quality or None
    main(a)

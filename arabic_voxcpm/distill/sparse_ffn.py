"""2:4 sparse base_lm FFN for the engine's sparse int8 kernels (issue #8, phase 2): prune, record the dense teacher's
targets, recover, export.

    python -m arabic_voxcpm.distill.sparse_ffn prune  --out D/pruned                   # SparseGPT 2:4, layer by layer
    python -m arabic_voxcpm.distill.sparse_ffn record --out D/targets                  # the dense model's targets
    python -m arabic_voxcpm.distill.sparse_ffn train  --pruned D/pruned --targets D/targets --out D/train
    python -m arabic_voxcpm.distill.sparse_ffn export --pruned D/pruned --ckpt D/train/best --out D/model

Data (the owner's choice, 2026-10-15): contexts of the v2 training selection (teacher-v2's 135 h: lahgtna, SADA,
synthetic; quality-filtered; the evaluation set's speakers and texts held out), packed as in training (70 % cloning
by reference). All 11 adapters are equally likely: an adapter's draws come half from its own dialects (the
evaluation's routing, train.adapters.ADAPTER_OF_DIALECT; ab takes the Jordanian ones, hindi_english has none), half
from all 17 dialects evenly. The teacher is the dense LM with the same adapter.

Only base_lm's FFN (gate, up, down of its 28 layers) changes; the adapters (LoRAs on q/k/v/o) and every other weight
stay as they are, so the result loads like the dense model, with the 11 adapters.

- prune: SparseGPT (Frantar & Alistarh 2023) with the 2:4 pattern along the input dimension. Layer by layer: the
  calibration batches' hidden states enter layer l with layers < l already pruned; gate and up (one Hessian of their
  common input) are pruned first, then down on the inputs the pruned gate / up give. Resumable per layer.
- record: per loss position (a target patch) of the training plan, the dense teacher's LocDiT conditioning mu, its
  stop logits, and the FSQ latent (tanh(in_proj(h)), before rounding) of the previous position when that one is
  audio. Shards of --shard-batches batches; resumable shard by shard.
- train: the pruned model with the masks fixed; only the kept FFN values train (fp32, 528M), everything else frozen.
  Loss = relative MSE of mu + --w-fsq x relative MSE of the FSQ latents + --w-stop x KL(teacher || student) of the
  stop decision. AdamW, cosine schedule; the last --val-batches batches of the plan are the validation set (best
  checkpoint). Resumable from out/last.
- export: the model directory: the dense-shaped weights with exact zeros (what the engine's 2:4 check reads), the
  source model's other files and adapters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import threading
import time
from collections import Counter, defaultdict
from queue import Queue

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_MODEL = "/workspace/models/VoxCPM2-Arabic-Dialects"
FFN = ("gate_proj", "up_proj", "down_proj")
MASKS = os.path.join("sparse24", "masks.safetensors")  # not at the top: the engine loads every top-level *.safetensors


# --------------------------------------------------------------------------------------------- 2:4 masks and solve
def kept_index(kept: torch.Tensor) -> torch.Tensor:
    """bool [out, in] with exactly 2 True in every group of 4 along in -> the kept positions [out, in/4, 2] (int64,
    ascending)."""
    out, inp = kept.shape
    k4 = kept.reshape(out, inp // 4, 4)
    assert bool((k4.sum(-1) == 2).all()), "not exactly 2 of every 4 kept"
    pos = torch.arange(4, device=kept.device)
    return ((~k4).to(torch.int64) * 4 + pos).argsort(-1)[..., :2].sort(-1).values


def pack_index(idx: torch.Tensor) -> torch.Tensor:
    """[out, G, 2] positions -> uint8 [out, G] (first | second << 2)."""
    return (idx[..., 0] | (idx[..., 1] << 2)).to(torch.uint8)


def unpack_index(packed: torch.Tensor) -> torch.Tensor:
    p = packed.to(torch.int64)
    return torch.stack([p & 3, p >> 2], -1)


@torch.no_grad()
def sparsegpt_24(w: torch.Tensor, h: torch.Tensor, blocksize: int = 128, percdamp: float = 0.01):
    """SparseGPT's 2:4 solve: (pruned weight fp32 [out, in], kept mask bool [out, in], estimated output error relative
    to the output's energy, sum over the calibration inputs of |W x|^2).

    w [out, in], h [in, in] = sum of x x^T over the calibration inputs. Columns are processed left to right in blocks;
    at every 4th column the 2 of the next 4 with the smallest w^2 / [H^-1]_jj^2 are pruned, and the pruning error is
    spread over the columns not yet processed (the inverse Hessian's Cholesky rows)."""
    assert blocksize % 4 == 0
    w = w.detach().clone().float()
    h = h.detach().clone().float()
    energy = float((w @ h * w).sum()) / 2  # |W x|^2 / 2, the losses' scale
    cols = w.shape[1]
    dead = torch.diag(h) == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    damp = percdamp * torch.mean(torch.diag(h))
    diag = torch.arange(cols, device=w.device)
    for attempt in range(6):
        try:
            hd = h.clone()
            hd[diag, diag] += damp * (10 ** attempt)
            hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(hd)), upper=True)
            break
        except torch.linalg.LinAlgError:
            if attempt == 5:
                raise
    pruned = torch.zeros_like(w, dtype=torch.bool)
    losses = torch.zeros(w.shape[0], device=w.device)
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        w1 = w[:, i1:i2].clone()
        q1 = torch.zeros_like(w1)
        err1 = torch.zeros_like(w1)
        hinv1 = hinv[i1:i2, i1:i2]
        m1 = torch.zeros_like(w1, dtype=torch.bool)
        d1 = torch.diag(hinv1)
        for i in range(i2 - i1):
            col = w1[:, i]
            d = hinv1[i, i]
            if i % 4 == 0:
                score = w1[:, i : i + 4] ** 2 / d1[i : i + 4].reshape(1, -1) ** 2
                m1.scatter_(1, i + torch.topk(score, 2, dim=1, largest=False)[1], True)
            q = col.clone()
            q[m1[:, i]] = 0
            q1[:, i] = q
            losses += (col - q) ** 2 / d**2 / 2
            e = (col - q) / d
            w1[:, i:] -= e.unsqueeze(1) @ hinv1[i, i:].unsqueeze(0)
            err1[:, i] = e
        w[:, i1:i2] = q1
        pruned[:, i1:i2] = m1
        w[:, i2:] -= err1 @ hinv[i1:i2, i2:]
    return w, ~pruned, float(losses.sum()) / max(energy, 1e-30)


class Sparse24Linear(nn.Module):
    """A bias-free linear layer whose weight is 2:4 sparse along its input: the kept values [out, in/4, 2] are the
    (fp32) parameter, their positions a fixed buffer; the dense weight is scattered for every call."""

    def __init__(self, weight: torch.Tensor, idx: torch.Tensor):
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.register_buffer("idx", pack_index(idx), persistent=False)
        self.vals = nn.Parameter(weight.detach().float().reshape(self.out_features, -1, 4).gather(-1, idx).contiguous())

    def dense(self, dtype=torch.bfloat16) -> torch.Tensor:
        out, g = self.idx.shape
        w = torch.zeros(out, g, 4, device=self.vals.device, dtype=dtype)
        return w.scatter(-1, unpack_index(self.idx), self.vals.to(dtype)).reshape(out, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.dense(x.dtype))


# ------------------------------------------------------------------------------------------------- contexts
def add_data_args(ap) -> None:
    ap.add_argument("--model", default=DEFAULT_MODEL, help="the dense model with adapters/ (the teacher, the source)")
    ap.add_argument("--heldout", default="/workspace/data/arabic/eval/v2/heldout.json")
    ap.add_argument("--corpora", default="lahgtna,sada,synth")
    ap.add_argument("--quality", default="/workspace/data/arabic/quality/scores.jsonl")
    ap.add_argument("--max-cer", type=float, default=0.10)
    ap.add_argument("--min-dnsmos", type=float, default=3.0)
    ap.add_argument("--p-ref", type=float, default=0.7)
    ap.add_argument("--budget", type=int, default=8000, help="padded positions per batch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")


def own_dialects(names: list[str]) -> dict[str, list[str]]:
    """Each adapter's dialects in the evaluation's routing; ab (a second Jordanian adapter) takes jordanian's."""
    from arabic_voxcpm.train.adapters import ADAPTER_OF_DIALECT

    own = {a: sorted(d for d, x in ADAPTER_OF_DIALECT.items() if x == a) for a in names}
    if "ab" in own and "jordanian" in own:
        own["ab"] = list(own["jordanian"])
    return own


def adapter_plan(rows: list[dict], ds, names: list[str], n_draws: int, budget: int, seed: int):
    """[(adapter, [(row, rep), ...]), ...]: n_draws clips, an equal share per adapter, half from its own dialects and
    half from all dialects (each dialect equally likely, its clips evenly), packed per adapter into token-budget
    batches; the batches shuffled. rep counts a clip's draws (another reference / window per draw)."""
    from arabic_voxcpm.distill.teacher_data import pack_batches

    rng = random.Random(seed)
    by_dialect = defaultdict(list)
    for i, r in enumerate(rows):
        by_dialect[r["dialect"]].append(i)
    every = sorted(by_dialect)
    own = {a: [d for d in ds_ if d in by_dialect] for a, ds_ in own_dialects(names).items()}
    reps = Counter()
    by = defaultdict(list)
    for k in range(n_draws):
        a = names[k % len(names)]
        pool = own[a] if own[a] and rng.random() < 0.5 else every
        g = rng.choice(by_dialect[rng.choice(pool)])
        by[a].append((g, reps[g]))
        reps[g] += 1
    plan = []
    for a in names:
        samples = by[a]
        for b in pack_batches([ds.length(g) for g, _ in samples], budget, random.Random(f"{seed}-{a}")):
            plan.append((a, [samples[i] for i in b]))
    rng.shuffle(plan)
    return plan


class Contexts:
    """The selection, one TrainSet over it (no tags: the adapters' models), and plans."""

    def __init__(self, args):
        from arabic_voxcpm.distill.teacher_data import PlainTokenizer
        from arabic_voxcpm.train import adapters as A
        from arabic_voxcpm.train import dataset as D

        corpora = tuple(args.corpora.split(","))
        rows = D.select_rows(corpora, heldout=args.heldout or None, quality=args.quality or None,
                             max_cer=args.max_cer, min_dnsmos=args.min_dnsmos)
        store = D.LatentStore(corpora)
        self.rows = sorted((r for r in rows if r["uid"] in store), key=lambda r: (r["source"], r["uid"]))
        self.names = sorted(A.find(args.model))
        assert len(self.names) == 11, f"expected the 11 adapters, found {self.names}"
        self.ds = D.TrainSet(self.rows, store, PlainTokenizer(args.model), p_ref=args.p_ref, p_no_tag=0.0,
                             seed=args.seed, tagged=False)
        self.avg_patches = sum(r["frames"] for r in self.rows) / len(self.rows) / 4
        self.args = args

    def plan(self, patches: int, seed: int):
        return adapter_plan(self.rows, self.ds, self.names, int(patches / self.avg_patches) + 1, self.args.budget,
                            seed)

    def batch(self, samples, device) -> dict:
        from arabic_voxcpm.train import pack

        seqs = [self.ds.sample(g, rep)[0] for g, rep in samples]
        return {k: v.to(device, non_blocking=True) for k, v in pack.collate(seqs).items()}


def fingerprint(plan, extra: dict) -> str:
    h = hashlib.sha1(json.dumps(extra, sort_keys=True).encode())
    for a, b in plan:
        h.update(f"{a}:{b}".encode())
    return h.hexdigest()


def load_model(model_dir: str, device, resident: bool = True):
    """The VoxCPM2 of model_dir (bf16, eval, no AudioVAE) with its adapters' layout and the bank of all of them
    (resident on the device: ~0.8 GB, switching is by reference)."""
    from voxcpm.model.voxcpm2 import VoxCPM2Model

    from arabic_voxcpm.train import adapters as A

    adapters = A.find(model_dir)
    model = VoxCPM2Model.from_local(model_dir, optimize=False, device=str(device))
    model.audio_vae = None
    A.inject_model(model, A.shared_config(adapters))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    bank = A.AdapterBank(adapters, device=device, dtype=torch.bfloat16, resident=resident)
    return model, bank


def base_inputs(model, batch):
    """base_lm's input embeddings [B, T, H], the encoder embeddings at audio positions, and the masks (as
    train.forward.dit_inputs)."""
    dtype = model._dtype()
    dev = model.device
    text_tokens = batch["text_tokens"].to(dev, dtype=torch.long)
    text_mask = batch["text_mask"].to(dev, dtype=dtype)
    audio_feats = batch["audio_feats"].to(dev, dtype=dtype)
    am = batch["audio_mask"].to(dev).bool()
    B, T, P, D = audio_feats.shape
    enc = model.enc_to_lm_proj(model.feat_encoder(audio_feats[am].unsqueeze(0)))[0]
    feat_embed = torch.zeros(B, T, enc.shape[-1], device=dev, dtype=enc.dtype)
    feat_embed[am] = enc
    scale_emb = getattr(model.config.lm_config, "scale_emb", 1.0)
    if not getattr(model.config.lm_config, "use_mup", False):
        scale_emb = 1.0
    text_embed = model.base_lm.embed_tokens(text_tokens) * scale_emb
    amf = am.to(dtype).unsqueeze(-1)
    combined = text_mask.unsqueeze(-1) * text_embed + amf * feat_embed
    return combined, feat_embed, text_mask, am


def _shift(x: torch.Tensor) -> torch.Tensor:
    return torch.cat((torch.zeros_like(x[:, :1]), x[:, :-1]), dim=1)


def lm_outputs(model, batch):
    """At every loss position: (mu [M, C], stop logits [M, 2], FSQ latent of the previous position [M, L], whether the
    previous position is audio [M]). The FSQ rounding passes gradients straight through (its values are the
    inference ones). Equals train.forward.dit_inputs' mu (tests/test_sparse_ffn.py)."""
    dtype = model._dtype()
    with torch.no_grad():
        combined, feat_embed, text_mask, am = base_inputs(model, batch)
    lm_mask = batch["loss_mask"].to(model.device).bool()
    amf = am.to(dtype).unsqueeze(-1)
    raw, _ = model.base_lm(inputs_embeds=combined, is_causal=True)
    raw = raw.to(dtype)
    fsq = model.fsq_layer
    latent = torch.tanh(fsq.in_proj(raw))
    q = latent + (torch.round(latent * fsq.scale) / fsq.scale - latent).detach()
    enc_out = fsq.out_proj(q) * amf + raw * text_mask.unsqueeze(-1)
    lm_hidden = _shift(enc_out)
    res_in = model.fusion_concat_proj(torch.cat((enc_out, amf * feat_embed), dim=-1))
    res_out, _ = model.residual_lm(inputs_embeds=res_in, is_causal=True)
    res_hidden = _shift(res_out.to(dtype))
    h = lm_hidden[lm_mask]
    mu = torch.cat((model.lm_to_dit_proj(h), model.res_to_dit_proj(res_hidden[lm_mask])), dim=-1)
    stop = model.stop_head(model.stop_actn(model.stop_proj(h)))
    return mu, stop, _shift(latent)[lm_mask], _shift(am)[lm_mask]


# ------------------------------------------------------------------------------------------------- prune
def prune(args) -> None:
    from safetensors.torch import load_file, save_file

    dev = torch.device(args.device)
    os.makedirs(os.path.join(args.out, "layers"), exist_ok=True)
    ctx = Contexts(args)
    plan = ctx.plan(args.calib_patches, args.seed)[: args.calib_batches]
    print(f"calibration: {len(plan)} batches, adapters {dict(Counter(a for a, _ in plan))}", flush=True)
    model, bank = load_model(args.model, dev)
    lm = model.base_lm
    # base_lm's inputs for every calibration batch, the valid (unpadded) positions, the rotary embedding
    hs, valid, pos = [], [], []
    with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16):
        for a, samples in plan:
            bank.use(model, a)
            b = ctx.batch(samples, dev)
            combined, _, text_mask, am = base_inputs(model, b)
            hs.append(combined.to(torch.bfloat16))
            valid.append((text_mask.bool() | am).reshape(-1))
            pos.append(lm.rope_emb(torch.arange(combined.shape[1], device=dev)) if lm.rope_emb is not None else None)
    rows = int(sum(v.sum() for v in valid))
    print(f"calibration positions: {rows}", flush=True)
    stats = json.load(open(os.path.join(args.out, "prune.json"))) if os.path.exists(os.path.join(args.out, "prune.json")) else {}

    def run_layer(layer, hook_mod=None, collect=None):
        """Every batch through ``layer`` (its adapter on); with hook_mod, collect(x_valid) on its input."""
        handle = None
        if hook_mod is not None:
            state = {}

            def pre(mod, inp):
                x = inp[0].reshape(-1, inp[0].shape[-1])
                collect(x[state["valid"]].float())

            handle = hook_mod.register_forward_pre_hook(pre)
        outs = []
        try:
            with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16):
                for (a, _), x, v, p in zip(plan, hs, valid, pos):
                    bank.use(model, a)
                    if hook_mod is not None:
                        state["valid"] = v
                    outs.append(layer(x, p, True)[0].to(torch.bfloat16))
        finally:
            if handle is not None:
                handle.remove()
        return outs

    def hessian(layer, mod, dim):
        h = torch.zeros(dim, dim, device=dev, dtype=torch.float32)

        def collect(x):
            h.addmm_(x.t(), x)

        run_layer(layer, mod, collect)
        return h

    t0 = time.time()
    for li, layer in enumerate(lm.layers):
        mlp = layer.mlp
        path = os.path.join(args.out, "layers", f"layer-{li:02d}.safetensors")
        if li in args.dense_layers:  # kept dense (the engine runs such a layer with its dense kernels)
            pass
        elif os.path.exists(path):
            saved = load_file(path, device=str(dev))
            for n in FFN:
                getattr(mlp, n).weight.data.copy_(saved[f"{n}.weight"])
        else:
            hidden = mlp.gate_proj.in_features
            h = hessian(layer, mlp.gate_proj, hidden)
            out = {}
            info = {}
            for n in ("gate_proj", "up_proj"):
                lin = getattr(mlp, n)
                w, kept, err = sparsegpt_24(lin.weight, h, args.blocksize, args.percdamp)
                lin.weight.data.copy_(w.to(lin.weight.dtype))
                out[f"{n}.weight"], out[f"{n}.mask"] = lin.weight.data.contiguous(), pack_index(kept_index(kept))
                info[n] = err
            h = hessian(layer, mlp.down_proj, mlp.down_proj.in_features)
            lin = mlp.down_proj
            w, kept, err = sparsegpt_24(lin.weight, h, args.blocksize, args.percdamp)
            lin.weight.data.copy_(w.to(lin.weight.dtype))
            out["down_proj.weight"], out["down_proj.mask"] = lin.weight.data.contiguous(), pack_index(kept_index(kept))
            info["down_proj"] = err
            del h
            save_file({k: v.cpu() for k, v in out.items()}, path + ".tmp")
            os.replace(path + ".tmp", path)
            stats[str(li)] = info
            json.dump(stats, open(os.path.join(args.out, "prune.json"), "w"), indent=1)
            print(f"layer {li}: SparseGPT relative error gate {info['gate_proj']:.3g} up {info['up_proj']:.3g} "
                  f"down {info['down_proj']:.3g} ({time.time() - t0:.0f} s)", flush=True)
        hs = run_layer(layer)  # this layer pruned: the next one's inputs
    # the model directory: the source's files, the pruned weights, the masks
    state = load_file(os.path.join(args.model, "model.safetensors"))
    masks = {}
    for li in range(len(lm.layers)):
        if li in args.dense_layers:
            continue
        saved = load_file(os.path.join(args.out, "layers", f"layer-{li:02d}.safetensors"))
        for n in FFN:
            key = f"base_lm.layers.{li}.mlp.{n}.weight"
            assert state[key].shape == saved[f"{n}.weight"].shape
            state[key] = saved[f"{n}.weight"].to(state[key].dtype).contiguous()
            masks[key] = saved[f"{n}.mask"]
    write_model_dir(args.model, args.out, state, masks)
    json.dump({"source": os.path.realpath(args.model)}, open(os.path.join(args.out, "source.json"), "w"))
    print(f"pruned {len(masks)} FFN weights -> {args.out}", flush=True)


def write_model_dir(src: str, out: str, state: dict, masks: dict | None, link: bool = True) -> None:
    """out: model.safetensors (state), the masks (unless None), and the source model's other files and adapters
    (symlinked, or copied)."""
    from safetensors.torch import save_file

    os.makedirs(out, exist_ok=True)
    save_file(state, os.path.join(out, "model.safetensors.tmp"), metadata={"format": "pt"})
    os.replace(os.path.join(out, "model.safetensors.tmp"), os.path.join(out, "model.safetensors"))
    if masks is not None:
        os.makedirs(os.path.dirname(os.path.join(out, MASKS)), exist_ok=True)
        save_file(masks, os.path.join(out, MASKS))
    for name in os.listdir(src):
        if name in ("model.safetensors", "README.md", "sparse24", "layers", "prune.json", "source.json") \
                or name.startswith("."):
            continue
        s, d = os.path.join(src, name), os.path.join(out, name)
        if os.path.lexists(d):
            continue
        if link:
            os.symlink(os.path.realpath(s) if os.path.isfile(s) else os.path.abspath(s), d)
        elif os.path.isdir(s):
            shutil.copytree(s, d, symlinks=False)
        else:
            shutil.copy(os.path.realpath(s), d)


def derive(args) -> None:
    """A pruned directory whose last base_lm layers keep their dense FFN, made from a full prune without the GPU: the
    pruning is sequential (layer l's calibration inputs come through the already pruned layers < l), so layers before
    the first dense one are exactly what a prune with --dense-layers gives. Only a suffix of layers can be dense this
    way (a later layer was pruned on the inputs of a pruned earlier one)."""
    from safetensors.torch import load_file

    masks = load_file(os.path.join(args.pruned, MASKS))
    n_layers = 1 + max(int(k.split(".")[2]) for k in masks)
    dense = sorted(set(args.dense_layers))
    if not dense or dense != list(range(dense[0], n_layers)):
        raise SystemExit(f"--dense-layers must be the last layers (a suffix of 0..{n_layers - 1}), got {dense}")
    source = json.load(open(os.path.join(args.pruned, "source.json")))["source"]
    state = load_file(os.path.join(source, "model.safetensors"))
    pruned = load_file(os.path.join(args.pruned, "model.safetensors"))
    keep = {}
    for key, m in masks.items():
        if int(key.split(".")[2]) not in dense:
            state[key] = pruned[key]
            keep[key] = m
    write_model_dir(source, args.out, state, keep)
    stats_path = os.path.join(args.pruned, "prune.json")
    if os.path.exists(stats_path):
        stats = {k: v for k, v in json.load(open(stats_path)).items() if int(k) not in dense}
        json.dump(stats, open(os.path.join(args.out, "prune.json"), "w"), indent=1)
    json.dump({"source": source, "derived_from": os.path.realpath(args.pruned), "dense_layers": dense},
              open(os.path.join(args.out, "source.json"), "w"))
    print(f"{len(keep)} FFN weights 2:4 sparse, layers {dense} dense -> {args.out}", flush=True)


# ------------------------------------------------------------------------------------------------- record
TARGETS = ("mu", "stop", "fsq", "fsq_mask")


def record(args) -> None:
    dev = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    ctx = Contexts(args)
    plan = ctx.plan(args.patches, args.seed)
    config = {k: getattr(args, k) for k in ("model", "heldout", "corpora", "quality", "max_cer", "min_dnsmos", "p_ref",
                                            "budget", "seed", "patches", "shard_batches")}
    fp = fingerprint(plan, config)
    man_path = os.path.join(args.out, "manifest.json")
    man = json.load(open(man_path)) if os.path.exists(man_path) else None
    if man is not None and man["fingerprint"] != fp:
        raise SystemExit(f"{args.out} holds targets of another plan or configuration")
    n_shards = -(-len(plan) // args.shard_batches)
    man = man or {"fingerprint": fp, "config": config, "batches": len(plan), "shards": n_shards, "done": []}
    json.dump(man, open(man_path, "w"), indent=1)
    rows_per_batch = 0.0
    print(f"plan: {len(plan)} batches in {n_shards} shards, {len(man['done'])} done; adapters "
          f"{dict(Counter(a for a, _ in plan))}", flush=True)
    model, bank = load_model(args.model, dev)
    t0 = time.time()
    for s in range(n_shards):
        if s in man["done"]:
            continue
        shard = []
        for bi in range(s * args.shard_batches, min(len(plan), (s + 1) * args.shard_batches)):
            a, samples = plan[bi]
            bank.use(model, a)
            with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16):
                mu, stop, fsq, fsq_mask = lm_outputs(model, ctx.batch(samples, dev))
            shard.append({"batch": bi, "mu": mu.to(torch.bfloat16).cpu(), "stop": stop.float().cpu(),
                          "fsq": fsq.to(torch.bfloat16).cpu(), "fsq_mask": fsq_mask.cpu()})
            rows_per_batch = len(mu)
        path = os.path.join(args.out, f"shard-{s:05d}.pt")
        torch.save(shard, path + ".tmp")
        os.replace(path + ".tmp", path)
        man["done"].append(s)
        json.dump(man, open(man_path, "w"), indent=1)
        print(f"shard {s + 1}/{n_shards} ({rows_per_batch} rows in its last batch, {time.time() - t0:.0f} s)",
              flush=True)
    man["complete"] = True
    json.dump(man, open(man_path, "w"), indent=1)


# ------------------------------------------------------------------------------------------------- train
def sparsify(model, masks: dict) -> list[Sparse24Linear]:
    """base_lm's FFN linears -> Sparse24Linear with the recorded masks; returns them."""
    out = []
    for li, layer in enumerate(model.base_lm.layers):
        for n in FFN:
            lin = getattr(layer.mlp, n)
            key = f"base_lm.layers.{li}.mlp.{n}.weight"
            if key not in masks:  # a layer kept dense: frozen
                continue
            sp = Sparse24Linear(lin.weight, unpack_index(masks[key].to(lin.weight.device)))
            setattr(layer.mlp, n, sp)
            out.append(sp)
    return out


def losses(student, teacher: dict, w_fsq: float, w_stop: float) -> dict:
    mu, stop, fsq, fsq_mask = student
    mu_t = teacher["mu"].float()
    l_mu = F.mse_loss(mu.float(), mu_t) / mu_t.pow(2).mean().clamp(min=1e-8)
    m = teacher["fsq_mask"]
    f_t = teacher["fsq"].float()[m]
    l_fsq = F.mse_loss(fsq.float()[m], f_t) / f_t.pow(2).mean().clamp(min=1e-8) if m.any() else mu.new_zeros(())
    l_stop = F.kl_div(F.log_softmax(stop.float(), -1), F.log_softmax(teacher["stop"].float(), -1), log_target=True,
                      reduction="batchmean")
    return {"loss": l_mu + w_fsq * l_fsq + w_stop * l_stop, "mu": l_mu, "fsq": l_fsq, "stop": l_stop}


class Targets:
    """The recorded shards; one in memory at a time."""

    def __init__(self, path: str):
        self.path = path
        self.man = json.load(open(os.path.join(path, "manifest.json")))
        assert self.man.get("complete"), f"{path}: recording not complete"
        self.per = self.man["config"]["shard_batches"]
        self.cur, self.data = None, None

    def __getitem__(self, bi: int) -> dict:
        s = bi // self.per
        if s != self.cur:
            self.data = {d["batch"]: d for d in torch.load(os.path.join(self.path, f"shard-{s:05d}.pt"))}
            self.cur = s
        return self.data[bi]


def step_order(n_train: int, per: int, steps: int, seed: int) -> list[int]:
    """Training batch per step: epoch by epoch, shards in a random order, the batches of a shard in a random order
    (one shard loaded at a time)."""
    order, epoch = [], 0
    shards = sorted({b // per for b in range(n_train)})
    while len(order) < steps:
        rng = random.Random(f"{seed}-{epoch}")
        for s in rng.sample(shards, len(shards)):
            bs = [b for b in range(s * per, min(n_train, (s + 1) * per))]
            order += rng.sample(bs, len(bs))
        epoch += 1
    return order[:steps]


def save_ckpt(path: str, params: list[Sparse24Linear], opt, step: int, extra: dict, full: bool) -> None:
    from safetensors.torch import save_file

    tmp = path + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    save_file({f"{i}": p.vals.detach().float().cpu() if full else p.vals.detach().to(torch.bfloat16).cpu()
               for i, p in enumerate(params)}, os.path.join(tmp, "vals.safetensors"))
    if full:
        torch.save(opt.state_dict(), os.path.join(tmp, "optim.pt"))
    json.dump({"step": step, **extra}, open(os.path.join(tmp, "state.json"), "w"), indent=1)
    shutil.rmtree(path, ignore_errors=True)
    os.replace(tmp, path)


def load_vals(path: str, params: list[Sparse24Linear]) -> None:
    from safetensors.torch import load_file

    vals = load_file(os.path.join(path, "vals.safetensors"))
    for i, p in enumerate(params):
        p.vals.data.copy_(vals[f"{i}"].to(p.vals.dtype))


def train(args) -> None:
    from safetensors.torch import load_file

    from arabic_voxcpm.train.model import _checkpointed
    from voxcpm.modules.minicpm4.model import MiniCPMDecoderLayer

    dev = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    tg = Targets(args.targets)
    rec = tg.man["config"]
    for k in ("heldout", "corpora", "quality", "max_cer", "min_dnsmos", "p_ref", "budget", "seed"):
        setattr(args, k, rec[k])
    args.model = rec["model"]
    ctx = Contexts(args)
    plan = ctx.plan(rec["patches"], rec["seed"])
    if fingerprint(plan, rec) != tg.man["fingerprint"]:
        raise SystemExit("the training plan does not match the recorded targets")
    n_train = len(plan) - args.val_batches
    val = list(range(n_train, len(plan)))
    order = step_order(n_train, tg.per, args.steps, args.seed)

    model, bank = load_model(args.pruned, dev)
    params = sparsify(model, load_file(os.path.join(args.pruned, MASKS)))
    for p in params:
        p.vals.requires_grad = True
    for m in model.modules():
        if isinstance(m, MiniCPMDecoderLayer):
            _checkpointed(m)
    torch.cuda.empty_cache()
    opt = torch.optim.AdamW([p.vals for p in params], lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0, eps=1e-8,
                            fused=dev.type == "cuda")

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * t)))

    start, best = 0, float("inf")
    last = os.path.join(args.out, "last")
    if os.path.exists(os.path.join(last, "state.json")):
        load_vals(last, params)
        opt.load_state_dict(torch.load(os.path.join(last, "optim.pt"), map_location=dev))
        st = json.load(open(os.path.join(last, "state.json")))
        start, best = st["step"], st.get("best", best)
        print(f"resumed at step {start} (best validation {best:.5f})", flush=True)
    log = open(os.path.join(args.out, "log.jsonl"), "a")

    def evaluate() -> dict:
        tot = defaultdict(float)
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16):
            for bi in val:
                a, samples = plan[bi]
                bank.use(model, a)
                out = losses(lm_outputs(model, ctx.batch(samples, dev)), {k: v.to(dev) for k, v in tg[bi].items()
                                                                         if k in TARGETS}, args.w_fsq, args.w_stop)
                for k, v in out.items():
                    tot[k] += float(v) / len(val)
        return dict(tot)

    if start == 0:
        v0 = evaluate()
        print(f"validation before training: {json.dumps({k: round(x, 5) for k, x in v0.items()})}", flush=True)
        log.write(json.dumps({"step": 0, "val": v0}) + "\n")
        log.flush()

    # batches are built ahead in a thread (the targets of a shard stay in the main thread's cache)
    q: Queue = Queue(maxsize=3)

    def producer():
        for step in range(start, args.steps):
            a, samples = plan[order[step]]
            q.put((step, a, ctx.batch(samples, "cpu")))
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()
    t0, acc = time.time(), defaultdict(float)
    while (item := q.get()) is not None:
        step, a, batch = item
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        bank.use(model, a)
        teacher = {k: v.to(dev, non_blocking=True) for k, v in tg[order[step]].items() if k in TARGETS}
        batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            out = losses(lm_outputs(model, batch), teacher, args.w_fsq, args.w_stop)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        gn = torch.nn.utils.clip_grad_norm_([p.vals for p in params], args.clip)
        opt.step()
        for k, v in out.items():
            acc[k] += float(v)
        done = step + 1
        if done % args.log_every == 0:
            rec_ = {"step": done, "lr": lr_at(step), "grad_norm": float(gn), "s_per_step": (time.time() - t0) / args.log_every,
                    **{k: v / args.log_every for k, v in acc.items()},
                    "mem_gb": torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0}
            print(json.dumps(rec_), flush=True)
            log.write(json.dumps(rec_) + "\n")
            log.flush()
            t0, acc = time.time(), defaultdict(float)
        if done % args.eval_every == 0 or done == args.steps:
            v = evaluate()
            log.write(json.dumps({"step": done, "val": v}) + "\n")
            log.flush()
            print(f"step {done} validation: {json.dumps({k: round(x, 5) for k, x in v.items()})}", flush=True)
            if v["loss"] < best:
                best = v["loss"]
                save_ckpt(os.path.join(args.out, "best"), params, opt, done, {"val": v}, full=False)
        if done % args.save_every == 0 or done == args.steps:
            save_ckpt(last, params, opt, done, {"best": best}, full=True)
    open(os.path.join(args.out, "DONE"), "w").write(json.dumps({"steps": args.steps, "best": best}) + "\n")


# ------------------------------------------------------------------------------------------------- export
def export(args) -> None:
    from safetensors.torch import load_file

    pruned = args.pruned
    state = load_file(os.path.join(pruned, "model.safetensors"))
    masks = load_file(os.path.join(pruned, MASKS))
    if args.ckpt:
        vals = load_file(os.path.join(args.ckpt, "vals.safetensors"))
        keys = sorted(masks, key=lambda k: (int(k.split(".")[2]), FFN.index(k.split(".")[4])))  # sparsify's order
        assert len(vals) == len(keys), (len(vals), len(keys))
        for i, key in enumerate(keys):
            w = state[key]
            idx = unpack_index(masks[key])
            dense = torch.zeros(w.shape[0], w.shape[1] // 4, 4, dtype=w.dtype)
            state[key] = dense.scatter(-1, idx, vals[f"{i}"].to(w.dtype)).reshape(w.shape).contiguous()
    src = json.load(open(os.path.join(pruned, "source.json")))["source"]  # the dense model the pruning started from
    write_model_dir(src, args.out, state, None if args.copy else masks, link=not args.copy)
    if args.copy:  # a self-contained directory (no links, no masks), e.g. for an upload
        for root, dirs, files in os.walk(args.out):
            for name in dirs + files:
                assert not os.path.islink(os.path.join(root, name)), os.path.join(root, name)
    print(f"exported -> {args.out} (other files from {src})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prune")
    add_data_args(p)
    p.add_argument("--out", required=True)
    p.add_argument("--calib-patches", type=int, default=100_000, help="target patches drawn for the calibration plan")
    p.add_argument("--calib-batches", type=int, default=24)
    p.add_argument("--blocksize", type=int, default=128)
    p.add_argument("--percdamp", type=float, default=0.01)
    p.add_argument("--dense-layers", type=int, nargs="*", default=[], help="base_lm layers whose FFN stays dense")
    dv = sub.add_parser("derive")
    dv.add_argument("--pruned", required=True, help="a full prune's directory")
    dv.add_argument("--dense-layers", type=int, nargs="+", required=True)
    dv.add_argument("--out", required=True)
    r = sub.add_parser("record")
    add_data_args(r)
    r.add_argument("--out", required=True)
    r.add_argument("--patches", type=int, default=1_500_000, help="target patches (loss positions) of the plan")
    r.add_argument("--shard-batches", type=int, default=25)
    t = sub.add_parser("train")
    add_data_args(t)
    t.add_argument("--pruned", required=True)
    t.add_argument("--targets", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--steps", type=int, default=2000)
    t.add_argument("--lr", type=float, default=2e-5)
    t.add_argument("--warmup", type=int, default=100)
    t.add_argument("--min-lr-ratio", type=float, default=0.1)
    t.add_argument("--clip", type=float, default=1.0)
    t.add_argument("--w-fsq", type=float, default=0.5)
    t.add_argument("--w-stop", type=float, default=1.0)
    t.add_argument("--val-batches", type=int, default=8)
    t.add_argument("--log-every", type=int, default=25)
    t.add_argument("--eval-every", type=int, default=250)
    t.add_argument("--save-every", type=int, default=250)
    e = sub.add_parser("export")
    e.add_argument("--pruned", required=True)
    e.add_argument("--ckpt", default="", help="a train checkpoint (vals.safetensors); empty: the pruned weights")
    e.add_argument("--out", required=True)
    e.add_argument("--copy", action="store_true", help="copy the source files instead of linking them")
    a = ap.parse_args()
    {"prune": prune, "derive": derive, "record": record, "train": train, "export": export}[a.cmd](a)


if __name__ == "__main__":
    main()

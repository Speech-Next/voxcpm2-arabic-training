"""Reference augmentation (#5, teacher-v3): the voice-cloning reference is degraded, the target never, so that a noisy
reference still gives clean speech.

References are AudioVAE latents, so a reference is degraded in audio: the frozen AudioVAE decodes the reference window
(48 kHz), the waveform is band-limited to the encoder's 16 kHz, degraded, and encoded back. This runs on the GPU in the
training step, batched, under no_grad, before the forward (its memory is free again when the forward starts).

Per reference, drawn deterministically from the seed and the step (``draw``):
- reverberation (p_reverb): a synthetic room impulse response, exponentially decaying Gaussian noise with RT60
  0.2-0.8 s after a direct path, direct-to-reverberant ratio 0-12 dB (no RIR dataset);
- band-limiting (p_lowpass): a low-pass at 3-7 kHz (telephone, low-bitrate codecs);
- additive noise (p_noise, and whenever nothing else was drawn): coloured Gaussian noise, power spectrum 1 / f^beta
  with beta 0-2 (white to brown), at an SNR of 5-25 dB over the window.
The degraded window keeps the clean window's RMS level.

The round trip alone is not the identity: the latents of a clean decode -> encode differ from the original by 3 % of
the latent variance (MSE 0.054, cosine 0.98; decoder conditioned on 16 kHz, measured on synthetic-corpus windows),
against 51 % for a degraded window. Only the references that draw an augmentation take the round trip; the others stay
the original latents, which is what an inference engine encodes from a clean reference.

In the training step this costs about 1.2 s per 25 references (~120 s of audio, fp32, A10), about a quarter of a
teacher step at p 0.5. Precomputed instead, once (K degraded copies of every training clip; the step then takes the
reference window from one of them):

    python -m arabic_voxcpm.train.augment --out /workspace/data/arabic/refaug --k 2 --corpora lahgtna,sada,synth \
        --quality /workspace/data/arabic/quality/scores.jsonl --heldout /workspace/data/arabic/eval/v2/heldout.json
    # -> <out>/v<k>/<corpus>/latents/<shard>.pt, the corpora's shard layout; train.py --ref-aug-dir <out>
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import random
import zlib
from dataclasses import dataclass

import torch

SR = 16000  # the encoder's rate
HOP = 640  # encoder samples per latent frame


@dataclass
class AugConfig:
    p_noise: float = 0.8
    snr_db: tuple[float, float] = (5.0, 25.0)
    beta: tuple[float, float] = (0.0, 2.0)
    p_reverb: float = 0.5
    rt60: tuple[float, float] = (0.2, 0.8)
    drr_db: tuple[float, float] = (0.0, 12.0)
    p_lowpass: float = 0.2
    cutoff_hz: tuple[float, float] = (3000.0, 7000.0)


def draw(rng: random.Random, n: int, cfg: AugConfig = AugConfig()) -> dict:
    """Per-reference degradation parameters for n references (floats and flags, as lists)."""
    p = {k: [] for k in ("noise", "snr_db", "beta", "reverb", "rt60", "drr_db", "lowpass", "cutoff_hz", "seed")}
    for _ in range(n):
        reverb, lowpass = rng.random() < cfg.p_reverb, rng.random() < cfg.p_lowpass
        noise = rng.random() < cfg.p_noise or not (reverb or lowpass)
        p["noise"].append(noise)
        p["reverb"].append(reverb)
        p["lowpass"].append(lowpass)
        for k in ("snr_db", "beta", "rt60", "drr_db", "cutoff_hz"):
            p[k].append(rng.uniform(*getattr(cfg, k)))
        p["seed"].append(rng.getrandbits(62))
    return p


def _fft_len(n: int) -> int:
    return 1 << max(1, math.ceil(math.log2(n)))


def rms(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """[B, N], [B] -> [B] RMS over each row's first lengths samples."""
    mask = torch.arange(x.shape[1], device=x.device)[None] < lengths[:, None]
    return ((x * mask).pow(2).sum(1) / lengths.clamp(min=1)).sqrt()


def _randn(shape: tuple[int, ...], seeds: list[int], device) -> torch.Tensor:
    """One row of Gaussian noise per seed (each row its own generator: a reference's draw does not depend on the
    batch it is in)."""
    out = torch.empty(shape, device=device)
    g = torch.Generator(device=device)
    for i, s in enumerate(seeds):
        g.manual_seed(int(s))
        out[i] = torch.randn(shape[1:], generator=g, device=device)
    return out


def coloured_noise(n_rows: int, n: int, beta: torch.Tensor, seeds: list[int], device) -> torch.Tensor:
    """[B, n] unit-RMS Gaussian noise with power spectrum ~ 1 / f^beta (per row)."""
    w = _randn((n_rows, n), seeds, device)
    spec = torch.fft.rfft(w, dim=1)
    f = torch.fft.rfftfreq(n, 1 / SR, device=device).clamp(min=20.0)  # flat below 20 Hz: no DC blow-up
    spec = spec * (f[None] / 1000.0).pow(-beta[:, None] / 2)
    x = torch.fft.irfft(spec, n=n, dim=1)
    return x / x.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-8)


def rir(rt60: torch.Tensor, drr_db: torch.Tensor, seeds: list[int], device, max_rt60: float = 0.8) -> torch.Tensor:
    """[B, L] synthetic room impulse responses: a unit direct path, then from 2.5 ms on exponentially decaying
    Gaussian noise (-60 dB at rt60) with the energy that gives the direct-to-reverberant ratio drr_db."""
    L = int(max_rt60 * SR)
    t = torch.arange(L, device=device) / SR
    tail = _randn((len(seeds), L), [s ^ 0x5EED for s in seeds], device)
    tail = tail * torch.exp(-3 * math.log(10) * t[None] / rt60[:, None]) * (t[None] >= 0.0025)
    energy = tail.pow(2).sum(1, keepdim=True).clamp(min=1e-12)
    h = tail * (10 ** (-drr_db[:, None] / 10) / energy).sqrt()
    h[:, 0] = 1.0
    return h


def lowpass(x: torch.Tensor, cutoff_hz: torch.Tensor, width_hz: float = 500.0) -> torch.Tensor:
    """[B, N] low-passed per row (raised-cosine roll-off of width_hz around the cutoff), zero-padded FFT."""
    n = x.shape[1]
    m = _fft_len(n + 2048)
    spec = torch.fft.rfft(x, n=m, dim=1)
    f = torch.fft.rfftfreq(m, 1 / SR, device=x.device)
    r = ((f[None] - (cutoff_hz[:, None] - width_hz / 2)) / width_hz).clamp(0, 1)
    return torch.fft.irfft(spec * (0.5 * (1 + torch.cos(math.pi * r))), n=m, dim=1)[:, :n]


def degrade(wave: torch.Tensor, lengths: torch.Tensor, params: dict) -> torch.Tensor:
    """16 kHz waveforms [B, N] (row i valid up to lengths[i]) -> degraded copies, each at its clean RMS."""
    dev = wave.device
    B, N = wave.shape
    P = {k: torch.as_tensor(v, device=dev) for k, v in params.items() if k != "seed"}
    seeds = params["seed"]
    valid = (torch.arange(N, device=dev)[None] < lengths[:, None]).float()
    x = wave * valid
    level = rms(x, lengths)
    if bool(P["reverb"].any()):
        h = rir(P["rt60"].float(), P["drr_db"].float(), seeds, dev)
        m = _fft_len(N + h.shape[1])
        wet = torch.fft.irfft(torch.fft.rfft(x, n=m, dim=1) * torch.fft.rfft(h, n=m, dim=1), n=m, dim=1)[:, :N] * valid
        wet = wet * (level / rms(wet, lengths).clamp(min=1e-8))[:, None]
        x = torch.where(P["reverb"][:, None], wet, x)
    if bool(P["lowpass"].any()):
        x = torch.where(P["lowpass"][:, None], lowpass(x, P["cutoff_hz"].float()) * valid, x)
    if bool(P["noise"].any()):
        noise = coloured_noise(B, N, P["beta"].float(), seeds, dev)
        gain = rms(x, lengths) / 10 ** (P["snr_db"].float() / 20)
        x = x + torch.where(P["noise"][:, None], noise * gain[:, None], torch.zeros_like(noise)) * valid
    x = x * (level / rms(x, lengths).clamp(min=1e-8))[:, None]
    peak = x.abs().amax(1, keepdim=True)
    return x / peak.clamp(min=0.99) * 0.99  # only rows that would clip are scaled down


def to_16k(y: torch.Tensor) -> torch.Tensor:
    """48 kHz [B, 3N] -> 16 kHz [B, N]: ideal low-pass at 8 kHz and decimation (zero-padded FFT)."""
    n = y.shape[1] // 3
    m = _fft_len(y.shape[1] + 4096) * 3
    spec = torch.fft.rfft(y, n=m, dim=1)[:, : m // 6 + 1]
    return torch.fft.irfft(spec, n=m // 3, dim=1)[:, :n] / 3  # irfft normalizes by the 3x shorter length


class RefAugmenter:
    """Degrades the batch's marked references in place: batch["ref_aug"] [B, T] holds, at each augmented reference
    patch, its number of real frames (4, or fewer in the window's last patch), else 0."""

    def __init__(self, vae, cfg: AugConfig = AugConfig(), seed: int = 0, chunk_frames: int = 1250,
                 sr_cond: int | None = 16000):
        """chunk_frames: latent frames per round-trip batch (1250: ~1.8 GB, as fast as larger ones); sr_cond: the
        decoder's bandwidth condition (16000, the encoder's band: 40 % less round-trip drift than the default 48000)."""
        self.vae, self.cfg, self.seed, self.chunk_frames, self.sr_cond = vae, cfg, seed, chunk_frames, sr_cond

    @torch.no_grad()
    def roundtrip(self, windows: list[torch.Tensor], params: dict | None) -> list[torch.Tensor]:
        """Latent windows [F_i, 64] -> decoded, degraded (params; None: clean), encoded back: [F_i, 64] each."""
        dev = windows[0].device
        F = max(len(w) for w in windows)
        z = torch.zeros(len(windows), 64, F, device=dev)
        for i, w in enumerate(windows):
            z[i, :, : len(w)] = w.T.float()
        sr = None if self.sr_cond is None else torch.tensor([self.sr_cond], device=dev, dtype=torch.int32)
        wave = to_16k(self.vae.decode(z, sr)[:, 0].float())
        lengths = torch.tensor([len(w) * HOP for w in windows], device=dev)
        if params is not None:
            wave = degrade(wave, lengths, params)
        wave = wave * (torch.arange(wave.shape[1], device=dev)[None] < lengths[:, None])
        out = self.vae.encode(wave.unsqueeze(1), SR)  # [B, 64, F]
        return [out[i, :, : len(w)].T for i, w in enumerate(windows)]

    @torch.no_grad()
    def __call__(self, batch: dict, key: tuple) -> int:
        """Degrade batch["audio_feats"]'s marked references; key (e.g. (step, micro-batch)) seeds the draws.
        Returns the number of references degraded."""
        marks = batch["ref_aug"]
        rows = [b for b in range(marks.shape[0]) if bool(marks[b].any())]
        if not rows:
            return 0
        feats = batch["audio_feats"]  # [B, T, P, D]
        P, D = feats.shape[2], feats.shape[3]
        pos = [marks[b].nonzero()[:, 0] for b in rows]
        n_frames = [int(marks[b].sum()) for b in rows]
        windows = [feats[b, p].reshape(-1, D)[:n] for b, p, n in zip(rows, pos, n_frames)]
        params = draw(random.Random(hash((self.seed, 0x5EF, *key))), len(rows), self.cfg)  # ints only: stable hash
        for idx in chunks(n_frames, self.chunk_frames):
            new = self.roundtrip([windows[k] for k in idx], {k: [v[m] for m in idx] for k, v in params.items()})
            for k, z in zip(idx, new):
                b, p, n = rows[k], pos[k], n_frames[k]
                full = torch.zeros(len(p) * P, D, device=z.device, dtype=feats.dtype)
                full[:n] = z.to(feats.dtype)  # the window's padding frames stay zero
                feats[b, p] = full.reshape(len(p), P, D)
        return len(rows)


def chunks(lengths: list[int], budget: int) -> list[list[int]]:
    """Indices grouped by length, each group's padded size (count x longest) within budget (memory bound)."""
    out, cur = [], []
    for i in sorted(range(len(lengths)), key=lambda i: lengths[i]):
        if cur and lengths[i] * (len(cur) + 1) > budget:
            out.append(cur)
            cur = []
        cur.append(i)
    return out + [cur] if cur else out


def variant_dirs(root: str) -> list[str]:
    return sorted(d for d in glob.glob(os.path.join(root, "v*")) if os.path.isdir(d))


def precompute(args) -> None:
    """K degraded copies of every training clip, shard by shard (a shard written is skipped when run again)."""
    from arabic_voxcpm import vae as avae
    from arabic_voxcpm.data import manifest
    from arabic_voxcpm.data.store import Shard, ShardWriter
    from arabic_voxcpm.train import dataset as D

    corpora = tuple(args.corpora.split(","))
    keep = {r["uid"] for r in D.select_rows(corpora, heldout=args.heldout, quality=args.quality,
                                            max_cer=args.max_cer, min_dnsmos=args.min_dnsmos)}
    if args.max_gb:
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    aug = RefAugmenter(avae.load_vae(args.model), seed=args.seed, chunk_frames=args.chunk,
                       sr_cond=args.sr_cond or None)
    for c in corpora:
        for path in sorted(glob.glob(os.path.join(manifest.ROOT, c, "latents", "*.pt"))):
            shard = Shard(path)
            uids = [u for u in shard.index if u in keep]
            for k in range(args.k):
                out = os.path.join(args.out, f"v{k}", c, "latents", os.path.basename(path))
                if os.path.exists(out):
                    continue
                os.makedirs(os.path.dirname(out), exist_ok=True)
                w = ShardWriter(out)
                lat = [shard[u].to("cuda") for u in uids]
                for idx in chunks([len(z) for z in lat], args.chunk):
                    # each clip's draw depends on (seed, k, uid) only
                    ps = [draw(random.Random(zlib.crc32(f"{args.seed}:{k}:{uids[i]}".encode())), 1, aug.cfg) for i in idx]
                    params = {key: [p[key][0] for p in ps] for key in ps[0]}
                    for i, z in zip(idx, aug.roundtrip([lat[i] for i in idx], params)):
                        w.add(uids[i], z)
                w.close()
                print(f"{out}: {len(uids)} clips", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Precompute degraded reference variants (" + __doc__.split("\n")[0] + ")")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=2, help="variants per clip")
    ap.add_argument("--corpora", default="lahgtna,sada,synth")
    ap.add_argument("--model", default="/workspace/models/VoxCPM2-Arabic-Dialects")
    ap.add_argument("--quality", help="as train.py: only the clips training can use")
    ap.add_argument("--heldout", default="/workspace/data/arabic/eval/v2/heldout.json")
    ap.add_argument("--max-cer", type=float, default=0.10)
    ap.add_argument("--min-dnsmos", type=float, default=3.0)
    ap.add_argument("--chunk", type=int, default=1250, help="latent frames per round-trip batch (~1.8 GB)")
    ap.add_argument("--sr-cond", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-gb", type=float, default=6.0, help="cap this process's GPU memory (shared GPU)")
    precompute(ap.parse_args())

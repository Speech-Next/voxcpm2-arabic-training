"""Training sequences in VoxCPM2's layout (as the upstream packer builds them), from latents.

Tracks per position: text token, audio patch [P, D], text / audio / loss masks, stop label.
- plain:     [text, 101, audio, 102]               loss on audio
- reference: [103, ref, 104, text, 101, audio, 102] loss on audio only (voice cloning by timbre)
The stop label is 1 at the last audio position.
"""

from __future__ import annotations

import torch

AUDIO_START, AUDIO_END, REF_START, REF_END = 101, 102, 103, 104
PATCH = 4


def patches(latents: torch.Tensor) -> torch.Tensor:
    """[frames, D] -> [ceil(frames / 4), 4, D], zero-padded at the end."""
    pad = (-len(latents)) % PATCH
    if pad:
        latents = torch.cat([latents, latents.new_zeros(pad, latents.shape[1])])
    return latents.reshape(-1, PATCH, latents.shape[1])


def _seq(parts):
    """parts: (kind, value) with kind "tok" (list of ids, no loss), "ref" / "audio" ([n, P, D] patches)."""
    toks, feats, tmask, amask, lmask = [], [], [], [], []
    dim = next(v.shape[-1] for k, v in parts if k != "tok")
    for kind, v in parts:
        n = len(v)
        if kind == "tok":
            toks.append(torch.as_tensor(v, dtype=torch.int32))
            feats.append(torch.zeros(n, PATCH, dim))
            tmask.append(torch.ones(n, dtype=torch.int32))
            amask.append(torch.zeros(n, dtype=torch.int32))
            lmask.append(torch.zeros(n, dtype=torch.int32))
        else:
            toks.append(torch.zeros(n, dtype=torch.int32))
            feats.append(v.float())
            tmask.append(torch.zeros(n, dtype=torch.int32))
            amask.append(torch.ones(n, dtype=torch.int32))
            lmask.append(torch.full((n,), int(kind == "audio"), dtype=torch.int32))
    labels = torch.zeros(sum(len(t) for t in toks), dtype=torch.int32)
    labels[-2] = 1
    return {"text_tokens": torch.cat(toks), "audio_feats": torch.cat(feats), "text_mask": torch.cat(tmask),
            "audio_mask": torch.cat(amask), "loss_mask": torch.cat(lmask), "labels": labels}


def plain(text_ids: list[int], audio: torch.Tensor) -> dict:
    return _seq([("tok", list(text_ids) + [AUDIO_START]), ("audio", audio), ("tok", [AUDIO_END])])


def with_ref(ref: torch.Tensor, text_ids: list[int], audio: torch.Tensor) -> dict:
    return _seq([("tok", [REF_START]), ("ref", ref), ("tok", [REF_END] + list(text_ids) + [AUDIO_START]),
                 ("audio", audio), ("tok", [AUDIO_END])])


def collate(seqs: list[dict]) -> dict:
    """Right-pad to the longest sequence (causal attention: padding never reaches real positions)."""
    n = max(len(s["text_tokens"]) for s in seqs)
    out = {}
    for k in seqs[0]:
        rows = []
        for s in seqs:
            x = s[k]
            pad = n - len(x)
            rows.append(torch.cat([x, x.new_zeros((pad,) + tuple(x.shape[1:]))]) if pad else x)
        out[k] = torch.stack(rows)
    out["position_ids"] = torch.arange(n).expand(len(seqs), n).clone()
    return out

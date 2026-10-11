"""VoxCPM2's LocDiT sampler, as the inference engine runs it (nanovllm_voxcpm models/voxcpm2/model_utils_cfm.py):
Euler steps over the sway-sampled grid from t = 1 (noise) to 0, the first step skipped (zero init), classifier-free
guidance with CFG-zero* (the unconditional branch has mu = 0, rescaled by the optimized scale). The estimator is the
model's feat_decoder.estimator: estimator(x, mu, t, cond, dt) -> velocity; dt is 0 unless the model is mean-mode.
"""

from __future__ import annotations

import math

import torch


def t_span(steps: int, device=None) -> torch.Tensor:
    t = torch.linspace(1, 0, steps + 1, device=device, dtype=torch.float32)
    return t + (torch.cos(torch.pi / 2 * t) - 1 + t)


def zero_init_steps(span_len: int) -> int:
    return max(1, int(span_len * 0.04))


def _optimized_scale(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    dot = (pos * neg).sum(1, keepdim=True)
    return dot / ((neg**2).sum(1, keepdim=True) + 1e-8)


@torch.no_grad()
def solve(estimator, z: torch.Tensor, mu: torch.Tensor, cond: torch.Tensor, cfg_value: float = 2.0,
          steps: int = 10, mean_mode: bool = False) -> torch.Tensor:
    """z, cond: [B, D, P]; mu: [B, C]. Returns the solved patch [B, D, P] (float32)."""
    span = t_span(steps, z.device)
    x = z.float()
    b = x.shape[0]
    dtype = next(estimator.parameters()).dtype
    zero = zero_init_steps(len(span))
    t, dt = span[0], span[0] - span[1]
    for step in range(1, len(span)):
        if step > zero:
            x_in = torch.cat([x, x]).to(dtype)
            mu_in = torch.cat([mu, torch.zeros_like(mu)]).to(dtype)
            t_in = t.expand(2 * b).to(dtype)
            dt_in = (dt if mean_mode else torch.zeros_like(dt)).expand(2 * b).to(dtype)
            cond_in = torch.cat([cond, cond]).to(dtype)
            v = estimator(x_in, mu_in, t_in, cond_in, dt_in).float()
            pos, neg = v[:b], v[b:]
            s = _optimized_scale(pos.reshape(b, -1), neg.reshape(b, -1)).view(b, 1, 1)
            v = neg * s + cfg_value * (pos - neg * s)
            x = x - dt * v
        t = t - dt
        if step < len(span) - 1:
            dt = t - span[step + 1]
    return x

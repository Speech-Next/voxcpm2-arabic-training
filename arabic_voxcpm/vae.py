"""VoxCPM2's AudioVAE encoder: 16 kHz waveforms -> 25 Hz x 64 latents (the model's audio features).

The encoder is causal and deterministic (the posterior mean), so a batch padded on the right gives every clip the
latents it gets alone: ceil(samples / 640) frames, the last one seeing the same zero padding.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import torch

SAMPLE_RATE = 16000
HOP = 640  # samples per latent frame (25 Hz)


def load_vae(model_dir: str, device: str = "cuda"):
    from voxcpm.model.voxcpm2 import VoxCPMConfig
    from voxcpm.modules.audiovae import AudioVAEV2

    config = VoxCPMConfig.model_validate_json(open(os.path.join(model_dir, "config.json"), encoding="utf-8").read())
    vae = AudioVAEV2(config=config.audio_vae_config)
    state = torch.load(os.path.join(model_dir, "audiovae.pth"), map_location="cpu", weights_only=True)
    vae.load_state_dict(state.get("state_dict", state))
    assert vae.sample_rate == SAMPLE_RATE and vae.chunk_size == HOP
    return vae.to(device).eval()


def exact() -> None:
    """TF32 off. Then a clip's latents do not depend on the batch it is encoded in (tests/test_vae.py; on an A10 99.99 %
    of the fp16 values were the same across batch shapes, the rest 1 fp16 step apart). With TF32 (torch's default for
    convolutions) the algorithms change with the batch shape: 47 % of the fp16 values differ, 6e-4 relative.
    --restore encodes this way, so re-encoding some of a shard's clips gives what encoding all of them would."""
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def frames(samples: int) -> int:
    return math.ceil(samples / HOP)


@torch.inference_mode()
def encode(vae, waves: list[np.ndarray] | list[torch.Tensor]) -> list[torch.Tensor]:
    """Latents [frames, 64] (float32, on the VAE's device) of each 16 kHz mono waveform."""
    device = next(vae.parameters()).device
    lens = [len(w) for w in waves]
    batch = torch.zeros(len(waves), 1, frames(max(lens)) * HOP, device=device)
    for i, w in enumerate(waves):
        batch[i, 0, : lens[i]] = torch.as_tensor(w, dtype=torch.float32, device=device)
    z = vae.encode(batch, SAMPLE_RATE)  # [B, 64, T]
    return [z[i, :, : frames(n)].T for i, n in enumerate(lens)]

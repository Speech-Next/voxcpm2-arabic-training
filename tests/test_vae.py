"""The batched encoder gives each clip the latents it gets alone (GPU)."""

import os

import numpy as np
import pytest
import torch

from arabic_voxcpm import vae as avae

MODEL = os.environ.get("VOXCPM2_MODEL", "/workspace/models/VoxCPM2-Arabic-Dialects")


@pytest.mark.skipif(not torch.cuda.is_available() or not os.path.isdir(MODEL), reason="needs the GPU and the model")
def test_batch_equals_single():
    # TF32 convolutions pick other algorithms for other batch shapes (~2e-4 relative, below the float16 the latents
    # are stored in); without TF32 the batch must match exactly, padding included
    torch.backends.cudnn.allow_tf32 = False
    vae = avae.load_vae(MODEL)
    rng = np.random.default_rng(0)
    waves = [rng.standard_normal(n).astype(np.float32) * 0.1 for n in (16000, 23456, 40001, 640)]
    batched = avae.encode(vae, waves)
    for w, z in zip(waves, batched):
        alone = avae.encode(vae, [w])[0]
        assert z.shape == (avae.frames(len(w)), 64)
        torch.testing.assert_close(z, alone, rtol=1e-5, atol=1e-5)

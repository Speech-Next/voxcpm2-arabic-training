"""Our forward feeds the LMs, the stop head and the LocDiT loss exactly what upstream's does (GPU, the real model).

The LocDiT loss is replaced by a deterministic masked regression of its inputs, so that both forwards must agree on
the rows, values and normalization they pass to it."""

import os

import pytest
import torch

MODEL = "/workspace/models/VoxCPM2-Arabic-Dialects"


@pytest.mark.skipif(not torch.cuda.is_available() or not os.path.isdir(MODEL), reason="needs the GPU and the model")
def test_forward_matches_upstream():
    from arabic_voxcpm.train import model as M
    from arabic_voxcpm.train import pack
    from arabic_voxcpm.train.forward import forward

    torch.manual_seed(0)
    model, _ = M.load(MODEL, {"r": 4, "alpha": 4})
    model.eval()
    proj = torch.randn(2048, 64 * 4, device="cuda") / 50

    def fake_loss(x1, mu, cond=None, tgt_mask=None, progress=0.0):
        pred = (mu.float() @ proj).view(x1.shape) + 0.5 * cond.float()
        losses = ((pred - x1.float()) ** 2).mean(dim=1)  # [rows, P]
        m = tgt_mask.squeeze(1).float()
        return (losses * m).sum() / m.sum().clamp(min=1)

    model.feat_decoder.compute_loss = fake_loss
    seqs = [pack.plain([5, 6, 7], torch.randn(9, 4, 64)), pack.with_ref(torch.randn(6, 4, 64), [8, 9], torch.randn(4, 4, 64)),
            pack.plain([3] * 11, torch.randn(13, 4, 64))]
    b = {k: v.cuda() for k, v in pack.collate(seqs).items()}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ours = forward(model, **b)
        up = model(**{k: b[k] for k in ("text_tokens", "text_mask", "audio_feats", "audio_mask", "loss_mask",
                                       "position_ids", "labels")})
    for k in ("loss/diff", "loss/stop"):
        torch.testing.assert_close(ours[k].float(), up[k].float(), rtol=2e-2, atol=1e-3)

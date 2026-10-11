"""The LocDiT student trainer end to end on a toy estimator (CPU): the mse stage, then dmd from its checkpoint."""

import argparse
import json
import math

import torch

from arabic_voxcpm.distill import dit, sampler


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(8, 8) / 4)
        self.a = torch.nn.Parameter(torch.ones(1))

    def forward(self, x, mu, t, cond, dt):
        return self.a * torch.tanh(x + (mu @ self.w)[:, :, None] + t[:, None, None] + dt[:, None, None]) + 0.3 * cond


def args(tmp_path, stage, **kw):
    a = dict(teacher="toy", data=str(tmp_path / "tdata"), out=str(tmp_path / "dit"), stage=stage, init=None,
             schedule="1,0.7071,0", layers=0, iters=60, batch=32, lr=1e-2, lr_fake=1e-2, warmup=5, real=0.0,
             reg=0.25, fake_steps=2, cfg=2.0, val_shards=1, val_every=30, log_every=10, seed=0, device="cpu")
    a.update(kw)
    return argparse.Namespace(**a)


def test_mse_then_dmd(tmp_path, monkeypatch):
    torch.manual_seed(0)
    teacher = Toy()
    (tmp_path / "tdata").mkdir()
    for k in range(2):  # the teacher's 10-step guided samples for seeded noise
        z, mu, cond = torch.randn(64, 8, 4), torch.randn(64, 8), torch.randn(64, 8, 4)
        with torch.no_grad():
            x = sampler.solve(teacher, z, mu, cond, cfg_value=2.0, steps=10)
        torch.save({"mu": mu.half(), "cond": cond.half(), "z": z.half(), "x": x.half(), "target": x.half()},
                   tmp_path / "tdata" / f"shard-{k:05d}.pt")
    monkeypatch.setattr(dit, "load_estimator", lambda path, device: teacher)

    dit.main(args(tmp_path, "mse"))
    val = [r for r in map(json.loads, open(tmp_path / "dit" / "log-mse.jsonl")) if r.get("event") == "val"]
    assert val[-1]["val_mse_teacher"] < 0.8 * val[0]["val_mse_teacher"]  # the toy 2-step student: 0.35 -> 0.24
    ck = torch.load(tmp_path / "dit" / "mse-best.pt", weights_only=True)
    assert ck["mean_mode"] and ck["steps"] == 2 and set(ck["state_dict"]) == {"feat_decoder.estimator.w",
                                                                                "feat_decoder.estimator.a"}

    dit.main(args(tmp_path, "dmd", init=str(tmp_path / "dit" / "mse-best.pt"), lr=1e-3))
    log = list(map(json.loads, open(tmp_path / "dit" / "log-dmd.jsonl")))
    steps = [r for r in log if "loss_dmd" in r]
    assert len(steps) == 6 and all(math.isfinite(r[k]) for r in steps for k in ("loss_dmd", "loss_reg", "loss_fake"))
    val = [r for r in log if r.get("event") == "val"]
    # the dmd student starts where the mse student ended and stays near the teacher
    assert abs(val[0]["val_mse_teacher"] - ck["val_mse_teacher"]) < 1e-2
    assert val[-1]["val_mse_teacher"] < 1.5 * ck["val_mse_teacher"]
    assert torch.load(tmp_path / "dit" / "dmd-last.pt", weights_only=True)["step"] == 60

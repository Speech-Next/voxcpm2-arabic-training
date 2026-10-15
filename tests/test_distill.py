"""The LocDiT student trainer end to end on toy estimators (CPU): mse then dmd from its checkpoint; with adapters, the
mse, spectral and dmd stages, each batch part under its adapter's LoRA, saved in the engine's checkpoint format."""

import argparse
import copy
import json
import math
import os

import torch
from safetensors.torch import save_file

from arabic_voxcpm.distill import dit, sampler
from arabic_voxcpm.distill import teacher_data as T
from arabic_voxcpm.train import adapters as A


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(8, 8) / 4)
        self.a = torch.nn.Parameter(torch.ones(1))

    def forward(self, x, mu, t, cond, dt):
        return self.a * torch.tanh(x + (mu @ self.w)[:, :, None] + t[:, None, None] + dt[:, None, None]) + 0.3 * cond


class ToyLoRA(torch.nn.Module):
    """A toy LocDiT whose mu projection takes the adapters' LoRA (q_proj, as a LocDiT attention projection)."""

    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(8, 8)
        self.a = torch.nn.Parameter(torch.ones(1))

    def forward(self, x, mu, t, cond, dt):
        return self.a * torch.tanh(x + self.q_proj(mu)[:, :, None] + t[:, None, None] + dt[:, None, None]) + 0.3 * cond


class ToyVAE(torch.nn.Module):
    """decode: latents [B, 8, T] -> audio [B, 1, 1024 T]."""

    def __init__(self):
        super().__init__()
        self.up = torch.nn.ConvTranspose1d(8, 1, kernel_size=1024, stride=1024)

    def decode(self, z):
        return torch.tanh(self.up(z))


def args(tmp_path, stage, **kw):
    a = dict(teacher="toy", adapters=None, data=str(tmp_path / "tdata"), out=str(tmp_path / "dit"), stage=stage,
             init="teacher", schedule="1,0.7071,0", mean_mode=True, layers=0, iters=60, batch=32, micro=256,
             lr=1e-2, lr_fake=1e-2, warmup=5, real=0.0, spec_weight=1.0, mse_weight=1.0, spec_val_rows=64, reg=0.25,
             fake_steps=2, fake_init="teacher", cfg=2.0, val_rows=64, val_every=30, save_every=0, log_every=10,
             checkpointing=True, seed=0, device="cpu")
    a.update(kw)
    return argparse.Namespace(**a)


def test_mse_then_dmd(tmp_path, monkeypatch):
    torch.manual_seed(0)
    teacher = Toy()
    (tmp_path / "tdata").mkdir()
    for k in range(2):  # the teacher's 10-step guided samples for seeded noise (an old-format teacher data dir)
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


def toy_adapter_model(root, names=("a0", "a1"), r=2):
    """A model directory with adapters/<name>/ holding toy LocDiT LoRAs on q_proj."""
    out = {}
    for k, name in enumerate(names):
        g = torch.Generator().manual_seed(k + 1)
        path = os.path.join(root, "adapters", name)
        os.makedirs(path)
        save_file({"feat_decoder.estimator.q_proj.lora_A": torch.randn(r, 8, generator=g) / 2,
                   "feat_decoder.estimator.q_proj.lora_B": torch.randn(8, r, generator=g) / 2},
                  os.path.join(path, A.WEIGHTS))
        json.dump({"lora_config": {"enable_dit": True, "r": r, "alpha": 2 * r, "target_modules_dit": ["q_proj"]}},
                  open(os.path.join(path, A.CONFIG), "w"))
        out[name] = path
    return out


def adapter_teacher_data(out, teacher, dirs, batches=12, rows=32):
    """teacher_data.py's layout from a toy adapter teacher: batches of one adapter each (a0, a1, base)."""
    est = copy.deepcopy(teacher)
    A.inject_dit(est, dirs)
    bank = A.AdapterBank(dirs, prefix=dit.PREFIX, dtype=torch.float32)
    names = sorted(dirs)
    plan = [([*names, None][bi % 3], [(bi, 0)] * rows) for bi in range(batches)]

    def run_batch(bi, adapter, samples):
        g = torch.Generator().manual_seed(100 + bi)
        n = len(samples)
        z = torch.randn(n, 8, 4, generator=g).to(torch.bfloat16)
        mu, cond = torch.randn(n, 8, generator=g), torch.randn(n, 8, 4, generator=g)
        bank.use(est, adapter)
        with torch.no_grad():
            x = sampler.solve(est, z.float(), mu, cond, cfg_value=2.0, steps=10)
        lm_mask = torch.ones(n, 1, dtype=torch.bool)
        o = T.batch_rows(lm_mask, samples, [{"ref": None, "ref_start": 0, "ref_frames": 0}] * n,
                         [g for g, _ in samples], bi, names.index(adapter) if adapter else -1, 0)
        o.update(mu=mu.to(torch.bfloat16), cond=cond.to(torch.bfloat16), z=z, x=x.to(torch.bfloat16),
                 target=(x + 0.1 * torch.randn_like(x)).to(torch.bfloat16))
        return o

    T.generate(plan, T.Writer(out, {"fingerprint": "toy", "adapters": names}, shard_rows=4 * rows), run_batch, 10**9)


def test_engine_grid():
    grid = dit.engine_grid(dit.parse_schedule("1,0.7071,0"))
    assert grid == [(1.0, 0.29296875), (0.70703125, 0.70703125)]  # the engine's bf16 t span, stepped in bf16
    assert dit.engine_grid(dit.parse_schedule("1,0")) == [(1.0, 1.0)]


def test_stft_loss():
    a = torch.randn(3, 7680)
    assert float(dit.stft_loss(a, a)) == 0.0
    assert float(dit.stft_loss(a + 0.1 * torch.randn_like(a), a)) > 0.01


def test_adapter_stages(tmp_path, monkeypatch):
    torch.manual_seed(0)
    base = ToyLoRA()
    dirs = toy_adapter_model(tmp_path / "model")
    adapter_teacher_data(str(tmp_path / "tdata"), base, dirs)
    monkeypatch.setattr(dit, "load_estimator", lambda path, device: copy.deepcopy(base))
    monkeypatch.setattr(dit, "load_vae", lambda path, device: ToyVAE())
    common = dict(teacher=str(tmp_path / "model"), batch=48, micro=10, val_rows=128)

    dit.main(args(tmp_path, "mse", **common))
    log = list(map(json.loads, open(tmp_path / "dit" / "log-mse.jsonl")))
    assert [r for r in log if r.get("event") == "adapters"][0]["adapters"] == ["a0", "a1"]
    val = [r for r in log if r.get("event") == "val"]
    assert set(val[0]["val_mse_teacher_by_adapter"]) == {"a0", "a1", "base"}
    assert val[-1]["val_mse_teacher"] < 0.8 * val[0]["val_mse_teacher"]
    for a in ("a0", "a1", "base"):
        assert val[-1]["val_mse_teacher_by_adapter"][a] < val[0]["val_mse_teacher_by_adapter"][a], a
    ck = torch.load(tmp_path / "dit" / "mse-best.pt", weights_only=True)
    # the engine's dit_checkpoint: the estimator's own tensors (no LoRA), bf16, the grid it rebuilds the t span from
    assert set(ck["state_dict"]) == {"feat_decoder.estimator.q_proj.weight", "feat_decoder.estimator.q_proj.bias",
                                     "feat_decoder.estimator.a"}
    assert all(v.dtype == torch.bfloat16 for v in ck["state_dict"].values())
    assert ck["mean_mode"] is True and ck["steps"] == 2 and "dit_layers" not in ck
    t_span = [float(ck["schedule"][0][0])] + [float(t - dt) for t, dt in ck["schedule"]]  # runner.py
    assert [round(t, 4) for t in t_span] == [1.0, 0.7071, 0.0] and ck["adapters"] == ["a0", "a1"]
    # loaded as the engine does (the state into its LocDiT, the adapters' LoRAs its own), the student needs its
    # adapter: with every row routed to the base model it is further from the teacher
    student = copy.deepcopy(base)
    student.load_state_dict({k[len(dit.PREFIX):]: v.float() for k, v in ck["state_dict"].items()})
    A.inject_dit(student, dirs)
    bank = A.AdapterBank(dirs, prefix=dit.PREFIX, dtype=torch.float32)
    data = dit.TeacherData(str(tmp_path / "tdata"), 128, "cpu")
    grid = dit.engine_grid(ck["schedule"])
    right = dit.validate(student, data, grid, 64, lambda n: bank.use(student, n), True)
    wrong = dit.validate(student, data, grid, 64, lambda n: bank.use(student, None), True)
    assert right["val_mse_teacher"] < 0.5 * wrong["val_mse_teacher"]

    dit.main(args(tmp_path, "spectral", init=str(tmp_path / "dit" / "mse-best.pt"), lr=1e-3, iters=20, val_every=10,
                  **common))
    log = list(map(json.loads, open(tmp_path / "dit" / "log-spectral.jsonl")))
    steps = [r for r in log if "loss_spec" in r]
    assert steps and all(math.isfinite(r["loss_spec"]) and math.isfinite(r["loss_mse"]) for r in steps)
    val = [r for r in log if r.get("event") == "val"]
    assert "val_spec" in val[0] and val[-1]["val_spec"] <= val[0]["val_spec"] * 1.05
    assert os.path.exists(tmp_path / "dit" / "spectral-best.pt")

    for fake_steps in (0, 2):  # 0: the regression term and an untrained fake score (dit.py used to fail on it)
        out = str(tmp_path / f"dmd{fake_steps}")
        dit.main(args(tmp_path, "dmd", init=str(tmp_path / "dit" / "spectral-best.pt"), lr=1e-3, iters=20,
                      val_every=10, fake_steps=fake_steps, out=out, **common))
        log = list(map(json.loads, open(os.path.join(out, "log-dmd.jsonl"))))
        steps = [r for r in log if "loss_dmd" in r]
        assert len(steps) == 2 and all(math.isfinite(r["loss_dmd"]) and math.isfinite(r["loss_reg"]) for r in steps)
        assert all(("loss_fake" in r) == (fake_steps > 0) for r in steps)
        assert torch.load(os.path.join(out, "dmd-last.pt"), weights_only=True)["step"] == 20

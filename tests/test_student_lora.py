"""Per-adapter student LoRAs (distill/student_lora.py) end to end on toy estimators (CPU): the frozen student under each
adapter's trainable LoRA, mse -> spectral -> dmd, each adapter's LoRA saved as an adapter directory of LocDiT tensors."""

import argparse
import copy
import json
import math
import os

import torch
from safetensors import safe_open

from arabic_voxcpm.distill import dit
from arabic_voxcpm.distill import student_lora as S
from arabic_voxcpm.train import adapters as A
from test_distill import ToyLoRA, ToyVAE, adapter_teacher_data, toy_adapter_model


def args(tmp_path, stage, **kw):
    a = dict(teacher=str(tmp_path / "model"), data=str(tmp_path / "tdata"), student=str(tmp_path / "student.pt"),
             out=str(tmp_path / "slora"), stage=stage, init="adapters", adapters=None, iters=40, rows=24, micro=16,
             lr=2e-2, lr_fake=1e-3, lr_fake_lora=1e-2, warmup=5, spec_weight=1.0, mse_weight=1.0, spec_val_rows=32,
             reg=0.25, fake_steps=2, cfg=2.0, val_rows=48, val_batch=32, val_every=20, log_every=10,
             checkpointing=True, seed=0, device="cpu")
    a.update(kw)
    return argparse.Namespace(**a)


def lora_file(d):
    with safe_open(os.path.join(d, A.WEIGHTS), framework="pt", device="cpu") as f:
        return {k: f.get_tensor(k) for k in f.keys()}


def setup(tmp_path, monkeypatch):
    torch.manual_seed(0)
    base = ToyLoRA()
    dirs = toy_adapter_model(tmp_path / "model")
    adapter_teacher_data(str(tmp_path / "tdata"), base, dirs)
    student = copy.deepcopy(base)  # a "distilled" student: the teacher's LocDiT, perturbed
    with torch.no_grad():
        student.q_proj.weight.add_(0.3 * torch.randn_like(student.q_proj.weight))
        student.a.mul_(0.8)
    dit.save(student, str(tmp_path / "student.pt"), [(1.0, 0.2929), (0.7071, 0.7071)])
    monkeypatch.setattr(dit, "load_estimator", lambda path, device: copy.deepcopy(base))
    monkeypatch.setattr(dit, "load_vae", lambda path, device: ToyVAE())
    return base, dirs


def test_rows_per_adapter(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    data = S.AdapterRows(str(tmp_path / "tdata"), val_rows=48)
    # the base model's rows (adapter -1) are left out; every adapter keeps its last rows for validation
    assert data.names == ["a0", "a1"] and data.rows("val") == {"a0": 48, "a1": 48}
    assert data.rows("train") == {"a0": 128 - 48, "a1": 128 - 48}
    b = data.batch(5, gen=torch.Generator().manual_seed(0))
    assert set(b) == {"a0", "a1"} and all(len(x["mu"]) == 5 for x in b.values())


def test_stages(tmp_path, monkeypatch):
    base, dirs = setup(tmp_path, monkeypatch)
    S.main(args(tmp_path, "mse"))
    out = tmp_path / "slora"
    log = list(map(json.loads, open(out / "log-mse.jsonl")))
    setup_rec = [r for r in log if r.get("event") == "setup"][0]
    assert setup_rec["adapters"] == ["a0", "a1"] and setup_rec["mean_mode"] is True
    val = [r for r in log if r.get("event") == "val"]
    for a in ("a0", "a1"):  # each adapter's own LoRA brings the frozen student closer to its teacher
        assert val[-1][a]["val_mse_teacher"] < 0.8 * val[0][a]["val_mse_teacher"], a
    # the saved LoRAs: the adapter's own LocDiT tensors, same names and shapes, changed; its config kept
    for a in ("a0", "a1"):
        new, old = lora_file(out / "mse-best" / a), lora_file(dirs[a])
        assert set(new) == set(old) and all(new[k].shape == old[k].shape and new[k].dtype == torch.float32
                                            for k in new)
        assert any(not torch.equal(new[k], old[k].float()) for k in new)
        cfg = json.load(open(out / "mse-best" / a / A.CONFIG))
        assert cfg["lora_config"] == json.load(open(os.path.join(dirs[a], A.CONFIG)))["lora_config"]
        assert cfg["student_lora"]["stage"] == "mse" and cfg["student_lora"]["step"] > 0
    state = json.load(open(out / "mse-best" / "state.json"))
    assert set(state) == {"a0", "a1"}

    # loaded as the engine does (the adapter's LoRA tensors replaced), the student is closer to the teacher
    ck = dit.load_checkpoint(str(tmp_path / "student.pt"))
    data = S.AdapterRows(str(tmp_path / "tdata"), val_rows=48)
    grid = dit.engine_grid(ck["schedule"])
    errs = {}
    for name, src in (("old", dirs), ("new", {a: str(out / "mse-best" / a) for a in dirs})):
        st = copy.deepcopy(base)
        A.inject_dit(st, dirs)
        st.load_state_dict({k[len(dit.PREFIX):]: v.float() for k, v in ck["state_dict"].items()})
        errs[name] = S.validate(st, S.LoRASet(src), data, grid, 32, True)
        assert not any("lora" in k for k in st.state_dict())  # bound tensors stay out of the state
    for a in ("a0", "a1"):
        assert errs["new"][a]["val_mse_teacher"] < errs["old"][a]["val_mse_teacher"]

    S.main(args(tmp_path, "spectral", init=str(out / "mse-best"), lr=1e-3, iters=20, val_every=10))
    log = list(map(json.loads, open(out / "log-spectral.jsonl")))
    steps = [r for r in log if "loss_spec" in r]
    assert steps and all(math.isfinite(r["loss_spec"]) for r in steps)
    assert "val_spec" in [r for r in log if r.get("event") == "val"][0]["a0"]
    # a stage starts from the previous stage's LoRAs: its step-0 validation is the previous best's
    assert abs([r for r in log if r.get("event") == "val"][0]["a0"]["val_mse_teacher"]
               - errs["new"]["a0"]["val_mse_teacher"]) < 1e-4

    for fake_steps in (0, 2):
        o = str(tmp_path / f"dmd{fake_steps}")
        S.main(args(tmp_path, "dmd", init=str(out / "spectral-best"), lr=1e-3, iters=20, val_every=10,
                    fake_steps=fake_steps, out=o))
        log = list(map(json.loads, open(os.path.join(o, "log-dmd.jsonl"))))
        steps = [r for r in log if "loss_dmd" in r]
        assert len(steps) == 2 and all(math.isfinite(r["loss_dmd"]) and math.isfinite(r["loss_reg"]) for r in steps)
        assert all(("loss_fake" in r) == (fake_steps > 0) for r in steps)
        assert json.load(open(os.path.join(o, "dmd-last", "state.json")))["step"] == 20


def test_only_chosen_adapters(tmp_path, monkeypatch):
    _, dirs = setup(tmp_path, monkeypatch)
    S.main(args(tmp_path, "mse", adapters="a1", iters=10, val_every=10))
    assert sorted(os.listdir(tmp_path / "slora" / "mse-last")) == ["a1", "state.json"]

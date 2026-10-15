"""Teacher data (distill/teacher_data.py) on toys (CPU): shards resume where an interrupted run stopped, and the replay
check reproduces recorded patches with their adapter and catches rows that do not match."""

import json
import os

import pytest
import torch
from safetensors.torch import save_file

from arabic_voxcpm.distill import teacher_data as T
from arabic_voxcpm.train import adapters as A


def toy_rows(bi, adapter, samples):
    """A batch's rows as run_batch returns them: 3 patches per sample, values fixed by the batch."""
    b = len(samples)
    lm_mask = torch.zeros(b, 6, dtype=torch.bool)
    lm_mask[:, 2:5] = True
    infos = [{"ref": (k + 1) % b if k % 2 else None, "ref_start": k, "ref_frames": 10 * k} for k in range(b)]
    ids = [g for g, _ in samples]
    out = T.batch_rows(lm_mask, samples, infos, ids, bi, {"a": 0, None: -1}[adapter], seed=5)
    g = torch.Generator().manual_seed(bi)
    n = len(out["clip"])
    for k in T.VECTORS:
        out[k] = torch.randn(n, 2 if k == "mu" else 3, *(() if k == "mu" else (4,)), generator=g).to(torch.bfloat16)
    return out


def read_all(out):
    m = json.load(open(os.path.join(out, "manifest.json")))
    parts = [torch.load(os.path.join(out, s["file"]), weights_only=True) for s in m["shards"]]
    return m, {k: torch.cat([p[k] for p in parts]) for k in parts[0]}


def test_rows_and_shard_resume(tmp_path):
    plan = [("a" if bi % 3 else None, [(10 * bi + k, k % 2) for k in range(1 + bi % 4)]) for bi in range(12)]
    meta = {"fingerprint": T.fingerprint(plan, {"x": 1}), "adapters": ["a"]}
    full = T.Writer(str(tmp_path / "full"), meta, shard_rows=7)
    T.generate(plan, full, toy_rows, patches=10**9)
    m, ref = read_all(tmp_path / "full")
    assert m["complete"] and m["rows"] == sum(3 * len(b) for _, b in plan) == len(ref["mu"])
    assert [s["rows"] for s in m["shards"]][:-1] == [7] * (len(m["shards"]) - 1)
    # row metadata: clip, patch index within the clip, the reference's clip id, position in the plan
    first = plan[1][1]
    rows1 = (ref["batch"] == 1).nonzero().flatten()
    assert ref["clip"][rows1].tolist() == [g for g, _ in first for _ in range(3)]
    assert ref["patch"][rows1].tolist() == [0, 1, 2] * len(first)
    assert ref["rep"][rows1].tolist() == [r for _, r in first for _ in range(3)]
    assert (ref["adapter"][ref["batch"] % 3 == 0] == -1).all() and (ref["adapter"][ref["batch"] % 3 != 0] == 0).all()
    assert ref["seed"][rows1].unique().tolist() == [T.batch_seed(5, 1)]
    b3 = (ref["batch"] == 3).nonzero().flatten()  # 4 samples: sample 1 cloned sample 2's clip, sample 0 none
    assert ref["ref"][b3].tolist() == [-1] * 3 + [plan[3][1][2][0]] * 3 + [-1] * 3 + [plan[3][1][0][0]] * 3
    assert m["rows_by_adapter"] == {"-1": int((ref["adapter"] == -1).sum()), "0": int((ref["adapter"] == 0).sum())}

    calls, die_at = [], [8]

    class Killed(Exception):
        pass

    def dies(bi, adapter, samples):
        calls.append(bi)
        if bi == die_at[0]:
            raise Killed
        return toy_rows(bi, adapter, samples)

    with pytest.raises(Killed):
        T.generate(plan, T.Writer(str(tmp_path / "cut"), meta, shard_rows=7), dies, patches=10**9)
    m_cut = json.load(open(tmp_path / "cut" / "manifest.json"))
    assert not m_cut["complete"] and m_cut["shards"]
    resumed_at = m_cut["next"][0]
    calls.clear()
    die_at[0] = None
    T.generate(plan, T.Writer(str(tmp_path / "cut"), meta, shard_rows=7), dies, patches=10**9)
    assert calls[0] == resumed_at < 8  # the batch of the first unwritten row, then on
    m2, resumed = read_all(tmp_path / "cut")
    assert m2["complete"] and m2["rows"] == m["rows"]
    for k in ref:
        assert torch.equal(ref[k], resumed[k]), k
    with pytest.raises(SystemExit, match="another configuration"):
        T.Writer(str(tmp_path / "cut"), dict(meta, fingerprint="other"))


def test_patches_limit_and_estimate(tmp_path):
    plan = [(None, [(bi, 0), (bi + 100, 0)]) for bi in range(10)]
    w = T.Writer(str(tmp_path / "d"), {"fingerprint": "f", "adapters": []}, shard_rows=4)
    T.generate(plan, w, toy_rows, patches=13)
    assert 13 <= w.m["rows"] < 13 + 6  # stops after the batch that reaches the target
    assert T.bytes_per_row() == 2 * 2048 + 4 * 2 * 64 * 4 + 34  # 6178 B: ~6.2 GB per million rows


class ToyLoRA(torch.nn.Module):
    """A toy LocDiT whose mu projection takes the adapters' LoRA (q_proj, as a LocDiT attention projection)."""

    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(8, 8)
        self.a = torch.nn.Parameter(torch.ones(1))

    def forward(self, x, mu, t, cond, dt):
        return self.a * torch.tanh(x + self.q_proj(mu)[:, :, None] + t[:, None, None] + dt[:, None, None]) + 0.3 * cond


def toy_adapters(root, names=("a0", "a1"), r=2):
    out = {}
    for k, name in enumerate(names):
        g = torch.Generator().manual_seed(k + 1)
        path = os.path.join(root, name)
        os.makedirs(path)
        save_file({"feat_decoder.estimator.q_proj.lora_A": torch.randn(r, 8, generator=g) / 2,
                   "feat_decoder.estimator.q_proj.lora_B": torch.randn(8, r, generator=g) / 2},
                  os.path.join(path, A.WEIGHTS))
        json.dump({"lora_config": {"enable_dit": True, "r": r, "alpha": 2 * r, "target_modules_dit": ["q_proj"]}},
                  open(os.path.join(path, A.CONFIG), "w"))
        out[name] = path
    return out


def toy_teacher_data(tmp_path, est, bank, names):
    plan = [(names[bi % 3] if bi % 3 < 2 else None, [(bi, 0)] * 16) for bi in range(9)]
    index = {n: i for i, n in enumerate(names)}

    def run_batch(bi, adapter, samples):
        g = torch.Generator().manual_seed(bi)
        n = len(samples)
        z = torch.randn(n, 8, 4, generator=g).to(torch.bfloat16)
        mu, cond = torch.randn(n, 8, generator=g).to(torch.bfloat16), torch.randn(n, 8, 4, generator=g).to(torch.bfloat16)
        bank.use(est, adapter)
        x = T.solve_chunked(est, z.float(), mu.float(), cond.float(), 2.0, 10, chunk=5)
        lm_mask = torch.zeros(n, 1, dtype=torch.bool)
        lm_mask[:, 0] = True
        out = T.batch_rows(lm_mask, samples, [{"ref": None, "ref_start": 0, "ref_frames": 0}] * n,
                           [g for g, _ in samples], bi, index.get(adapter, -1), 0)
        out.update(mu=mu, cond=cond, z=z, x=x.to(torch.bfloat16), target=x.to(torch.bfloat16))
        return out

    out = str(tmp_path / "td")
    T.generate(plan, T.Writer(out, {"fingerprint": "f", "adapters": names}, shard_rows=50), run_batch, 10**9)
    return out


def test_replay_reproduces_with_the_adapter(tmp_path):
    torch.manual_seed(0)
    dirs = toy_adapters(tmp_path / "adapters")
    est = ToyLoRA()
    A.inject_dit(est, dirs)
    bank = A.AdapterBank(dirs, prefix="feat_decoder.estimator.", dtype=torch.float32)
    out = toy_teacher_data(tmp_path, est, bank, sorted(dirs))

    def solve(z, mu, cond):  # the generator's chunks
        return T.solve_chunked(est, z, mu, cond, 2.0, 10, chunk=5)

    res = T.replay(out, 100, lambda n: bank.use(est, n), solve, "cpu", tol=0.02)
    assert res["rows"] >= 100 and res["batches"] == res["rows"] // 16 and res["max"] < 0.01  # bf16 storage of x
    assert res["without_adapter_median"] > 10 * res["with_adapter_median"]

    m = json.load(open(os.path.join(out, "manifest.json")))
    m["adapters"] = m["adapters"][::-1]  # rows routed to the wrong adapter
    json.dump(m, open(os.path.join(out, "manifest.json"), "w"))
    with pytest.raises(AssertionError, match="differ"):
        T.replay(out, 100, lambda n: bank.use(est, n), solve, "cpu", tol=0.02)

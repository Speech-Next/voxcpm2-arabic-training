"""2:4 sparse base_lm FFN (issue #8, phase 2): SparseGPT's 2:4 solve, the masks, the trainable sparse layer, the plan,
the losses and the step order on the CPU; lm_outputs against train.forward on the real model (GPU)."""

import os
from collections import Counter

import pytest
import torch

from arabic_voxcpm.distill import sparse_ffn as S


def test_sparsegpt_24_keeps_2_of_4_and_beats_magnitude():
    torch.manual_seed(0)
    out, inp, n = 32, 64, 512
    x = torch.randn(n, inp) @ (torch.eye(inp) + 0.3 * torch.randn(inp, inp))  # correlated inputs
    w = torch.randn(out, inp)
    w_sp, kept, err = S.sparsegpt_24(w, x.t() @ x, blocksize=16)
    assert kept.reshape(out, -1, 4).sum(-1).eq(2).all()
    assert torch.equal(w_sp == 0, ~kept) or bool((w_sp[~kept] == 0).all())
    mag = w.reshape(out, -1, 4).abs().topk(2, -1).indices
    m = torch.zeros(out, inp // 4, 4, dtype=torch.bool).scatter_(-1, mag, True).reshape(out, inp)
    e_sp = ((w_sp - w) @ x.t()).pow(2).sum()
    e_mag = ((w * m - w) @ x.t()).pow(2).sum()
    assert e_sp < 0.8 * e_mag, (float(e_sp), float(e_mag))
    assert err > 0


def test_index_pack_round_trip():
    torch.manual_seed(1)
    kept = torch.zeros(8, 16, 4, dtype=torch.bool)
    for r in range(8):
        for g in range(16):
            kept[r, g, torch.randperm(4)[:2]] = True
    kept = kept.reshape(8, 64)
    idx = S.kept_index(kept)
    assert idx.shape == (8, 16, 2) and bool((idx[..., 0] < idx[..., 1]).all())
    assert torch.equal(S.unpack_index(S.pack_index(idx)), idx)
    back = torch.zeros(8, 16, 4, dtype=torch.bool).scatter_(-1, idx, True).reshape(8, 64)
    assert torch.equal(back, kept)


def test_sparse_linear_is_the_masked_weight_and_trains_only_kept_values():
    torch.manual_seed(2)
    w = torch.randn(12, 32)
    kept = torch.zeros(12, 8, 4, dtype=torch.bool)
    kept[..., 1] = kept[..., 2] = True
    kept = kept.reshape(12, 32)
    lin = S.Sparse24Linear(w * kept, S.kept_index(kept))
    assert torch.equal(lin.dense(torch.float32), w * kept)
    x = torch.randn(5, 32, requires_grad=True)
    y = lin(x)
    torch.testing.assert_close(y, x @ (w * kept).t())
    y.sum().backward()
    g_dense = (torch.ones(5, 12).t() @ x.detach())  # d(sum y)/dW
    torch.testing.assert_close(lin.vals.grad, g_dense.reshape(12, 8, 4)[..., 1:3])
    assert lin.vals.shape == (12, 8, 2) and lin.vals.dtype == torch.float32


def _rows():
    dialects = ["egyptian", "lebanese", "syrian", "msa", "gulf", "yemeni"]
    return [{"uid": f"u{i}", "dialect": dialects[i % len(dialects)], "frames": 40 + i % 7} for i in range(300)]


class _DS:
    def length(self, i):
        return 50 + i % 13


def test_adapter_plan_is_balanced_and_deterministic(monkeypatch):
    monkeypatch.setattr(S, "own_dialects", lambda names: {"egyptian": ["egyptian"], "jordanian": ["lebanese", "syrian"],
                                                          "ab": ["lebanese", "syrian"], "hindi_english": []})
    names = ["ab", "egyptian", "hindi_english", "jordanian"]
    rows = _rows()
    plan = S.adapter_plan(rows, _DS(), names, 4000, 600, seed=3)
    assert plan == S.adapter_plan(rows, _DS(), names, 4000, 600, seed=3)
    per = Counter()
    own = Counter()
    for a, samples in plan:
        per[a] += len(samples)
        own[a] += sum(rows[g]["dialect"] in {"egyptian": ["egyptian"], "jordanian": ["lebanese", "syrian"],
                                             "ab": ["lebanese", "syrian"]}.get(a, []) for g, _ in samples)
    assert set(per.values()) == {1000}  # an equal share per adapter
    # half the draws from the adapter's own dialects, plus its share of the even draws over all 6
    assert 0.5 + 0.5 / 6 - 0.06 < own["egyptian"] / 1000 < 0.5 + 0.5 / 6 + 0.06
    assert 0.5 + 0.5 * 2 / 6 - 0.06 < own["jordanian"] / 1000 < 0.5 + 0.5 * 2 / 6 + 0.06
    seen = Counter()
    for _, samples in plan:  # every draw of a clip has its own rep
        for g, rep in samples:
            seen[(g, rep)] += 1
    assert max(seen.values()) == 1


def test_losses_are_zero_for_the_teacher_itself():
    torch.manual_seed(4)
    mu, stop, fsq = torch.randn(10, 16), torch.randn(10, 2), torch.randn(10, 4)
    m = torch.rand(10) > 0.3
    out = S.losses((mu, stop, fsq, m), {"mu": mu, "stop": stop, "fsq": fsq, "fsq_mask": m}, 0.5, 1.0)
    for k in ("loss", "mu", "fsq", "stop"):
        assert abs(float(out[k])) < 1e-6, k
    out = S.losses((mu + 0.1, stop, fsq, m), {"mu": mu, "stop": stop, "fsq": fsq, "fsq_mask": m}, 0.5, 1.0)
    assert float(out["mu"]) > 0


def test_step_order_epochs_by_shard():
    order = S.step_order(n_train=23, per=5, steps=50, seed=0)
    assert len(order) == 50 and sorted(order[:23]) == list(range(23)) and sorted(order[23:46]) == list(range(23))
    shards = [b // 5 for b in order[:23]]  # a shard's batches are consecutive
    assert sum(1 for i in range(1, 23) if shards[i] != shards[i - 1]) == 4


def test_write_model_dir_links_and_copies(tmp_path):
    src = tmp_path / "src"
    (src / "adapters" / "a").mkdir(parents=True)
    (src / "adapters" / "a" / "lora_weights.safetensors").write_bytes(b"x")
    for f in ("config.json", "audiovae.pth", "model.safetensors", "README.md"):
        (src / f).write_text(f)
    state = {"w": torch.zeros(2, 4)}
    S.write_model_dir(str(src), str(tmp_path / "linked"), state, {"m": torch.zeros(2, 1, dtype=torch.uint8)})
    assert os.path.islink(tmp_path / "linked" / "config.json") and not (tmp_path / "linked" / "README.md").exists()
    assert (tmp_path / "linked" / S.MASKS).exists()
    S.write_model_dir(str(src), str(tmp_path / "copied"), state, None, link=False)
    assert not os.path.islink(tmp_path / "copied" / "adapters") and not (tmp_path / "copied" / S.MASKS).exists()
    assert (tmp_path / "copied" / "adapters" / "a" / "lora_weights.safetensors").read_bytes() == b"x"


MODEL = "/workspace/models/VoxCPM2-Arabic-Dialects"


@pytest.mark.skipif(not torch.cuda.is_available() or not os.path.isdir(MODEL), reason="needs the GPU and the model")
def test_lm_outputs_mu_matches_dit_inputs():
    from arabic_voxcpm.train import pack
    from arabic_voxcpm.train.forward import dit_inputs

    torch.manual_seed(0)
    model, bank = S.load_model(MODEL, "cuda")
    bank.use(model, "najdi")
    seqs = [pack.plain([5, 6, 7], torch.randn(9, 4, 64)), pack.with_ref(torch.randn(6, 4, 64), [8, 9], torch.randn(4, 4, 64))]
    b = {k: v.cuda() for k, v in pack.collate(seqs).items()}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        mu, stop, fsq, prev_audio = S.lm_outputs(model, b)
        mu_ref, *_ , lm_mask = dit_inputs(model, **b)
    torch.testing.assert_close(mu.float(), mu_ref.float(), rtol=0, atol=0)
    assert stop.shape == (len(mu), 2) and fsq.shape[0] == len(mu) and prev_audio.dtype == torch.bool
    assert int(prev_audio.sum()) == int(lm_mask.sum()) - 2  # the first target patch of each sequence follows text

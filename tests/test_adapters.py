"""The dialect adapters on the PyTorch model (train/adapters.py): the LoRA math against the inference engine's loader
and upstream voxcpm's LoRALinear, switching adapters, and the routing of clips to adapters (CPU)."""

import ast
import dataclasses
import glob
import importlib.util
import json
import os
import sys
import types

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

from arabic_voxcpm.distill import teacher_data as T
from arabic_voxcpm.eval.build import CURRENT_ADAPTER, DIALECT_OF_GROUP
from arabic_voxcpm.train import adapters as A
from arabic_voxcpm.train import dialects

ENGINE = os.environ.get("ENGINE", "/workspace/voxcpm2-arabic-dialects-turbo")
ADAPTERS_MODEL = glob.glob("/workspace/.hf_home/hub/models--VoxCPM-Kernel--VoxCPM2-Arabic-Dialects/snapshots/*")
ATTN = "feat_decoder.estimator.decoder.layers.0.self_attn"
SIZES = {"q_proj": (12, 16), "k_proj": (12, 4), "v_proj": (12, 4), "o_proj": (16, 12)}  # (in, out)


def write_adapter(path, r=4, alpha=8, seed=0, prefix=ATTN, sizes=SIZES):
    g = torch.Generator().manual_seed(seed)
    t = {}
    for name, (i, o) in sizes.items():
        t[f"{prefix}.{name}.lora_A"] = torch.randn(r, i, generator=g)
        t[f"{prefix}.{name}.lora_B"] = torch.randn(o, r, generator=g)
    os.makedirs(path, exist_ok=True)
    save_file(t, os.path.join(path, A.WEIGHTS))
    json.dump({"base_model": "model", "lora_config": {
        "enable_lm": False, "enable_dit": True, "enable_proj": False, "r": r, "alpha": alpha, "dropout": 0.0,
        "target_modules_lm": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "target_modules_dit": ["q_proj", "v_proj", "k_proj", "o_proj"]}}, open(os.path.join(path, A.CONFIG), "w"))
    return t


def engine_loader():
    """The engine's lora_loader module alone: its two package imports (the engine's payload dataclasses, the model's
    packed_modules_mapping, read from the engine's model.py) stubbed, so it loads without flash-attn / triton."""
    path = os.path.join(ENGINE, "nanovllm_voxcpm/models/voxcpm2/lora_loader.py")
    model_py = os.path.join(ENGINE, "nanovllm_voxcpm/models/voxcpm2/model.py")
    if not (os.path.exists(path) and os.path.exists(model_py)):
        pytest.skip("inference library not available")
    mapping = None
    for node in ast.walk(ast.parse(open(model_py).read())):
        if isinstance(node, ast.ClassDef) and node.name == "VoxCPM2Model":
            for st in node.body:
                if isinstance(st, ast.Assign) and getattr(st.targets[0], "id", "") == "packed_modules_mapping":
                    mapping = ast.literal_eval(st.value)
    assert mapping

    @dataclasses.dataclass
    class LoRAModulePayload:
        lora_a: torch.Tensor
        lora_b: object
        effective_rank: int
        scaling: float

    @dataclasses.dataclass
    class LoRAModelPayload:
        modules: dict
        rank: int
        alpha: float

    stubs = {"nanovllm_voxcpm.engine.lora_manager": dict(LoRAModulePayload=LoRAModulePayload,
                                                         LoRAModelPayload=LoRAModelPayload),
             "nanovllm_voxcpm.models.voxcpm2.model": dict(VoxCPM2Model=type("VoxCPM2Model", (), {
                 "packed_modules_mapping": mapping}))}
    saved = {k: sys.modules.get(k) for k in stubs}
    try:
        for k, attrs in stubs.items():
            m = types.ModuleType(k)
            m.__dict__.update(attrs)
            sys.modules[k] = m
        spec = importlib.util.spec_from_file_location("engine_lora_loader", path)
        E = importlib.util.module_from_spec(spec)
        sys.modules["engine_lora_loader"] = E  # its dataclasses look their module up
        spec.loader.exec_module(E)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return E


def test_delta_matches_engine_loader(tmp_path):
    E = engine_loader()
    t = write_adapter(tmp_path / "a", r=4, alpha=8)
    payload = E.load_voxcpm2_lora_checkpoint(str(tmp_path / "a"))
    cfg = A.read_config(str(tmp_path / "a"))
    s = A.scaling(cfg)
    assert s == 2.0 and payload.alpha == 8 and payload.rank == 4
    qkv = payload.modules[f"{ATTN}.qkv_proj"]  # the engine packs q, k, v; B is scaled when a slot loads it
    assert qkv.scaling == s
    for i, name in enumerate(["q_proj", "k_proj", "v_proj"]):
        engine = (qkv.lora_b[i] * qkv.scaling) @ qkv.lora_a[i]
        torch.testing.assert_close(A.delta(t[f"{ATTN}.{name}.lora_A"], t[f"{ATTN}.{name}.lora_B"], s), engine)
    o = payload.modules[f"{ATTN}.o_proj"]
    torch.testing.assert_close(A.delta(t[f"{ATTN}.o_proj.lora_A"], t[f"{ATTN}.o_proj.lora_B"], s),
                               (o.lora_b * o.scaling) @ o.lora_a)


def toy_attn():
    """A bare estimator's module tree down to one attention block (decoder.layers.0.self_attn)."""
    root = nn.Module()
    root.decoder = nn.Module()
    root.decoder.layers = nn.ModuleList([nn.Module()])
    attn = root.decoder.layers[0].self_attn = nn.Module()
    for name, (i, o) in SIZES.items():
        attn.add_module(name, nn.Linear(i, o, bias=name != "o_proj"))
    attn.gate_proj = nn.Linear(12, 12)  # not a target
    return root


def test_adapter_linear_is_merged_weight(tmp_path):
    t = write_adapter(tmp_path / "a", r=4, alpha=8)
    root = toy_attn()
    base = {n: m.weight.detach().clone() for n, m in root.named_modules() if isinstance(m, nn.Linear)}
    assert A.inject_dit(root, {"a": str(tmp_path / "a")}) == 4
    assert sorted(n for n, m in root.named_modules() if isinstance(m, A.AdapterLinear)) == \
        sorted(f"decoder.layers.0.self_attn.{n}" for n in SIZES)
    sd_keys = set(root.state_dict())
    assert not any("lora" in k for k in sd_keys)  # the state stays nn.Linear's (engine checkpoint keys)
    bank = A.AdapterBank({"a": str(tmp_path / "a")}, prefix="feat_decoder.estimator.", dtype=torch.float32)
    bank.use(root, "a")
    x = torch.randn(3, 5, 12)
    for name in SIZES:
        m = root.get_submodule(f"decoder.layers.0.self_attn.{name}")
        key = f"{ATTN}.{name}"
        w = base[f"decoder.layers.0.self_attn.{name}"] + A.delta(t[key + ".lora_A"], t[key + ".lora_B"], 2.0)
        xin = x if name != "o_proj" else torch.randn(3, 5, 16)
        torch.testing.assert_close(m(xin), F.linear(xin, w, m.bias), rtol=1e-5, atol=1e-4)
    bank.use(root, None)  # the base model again
    m = root.get_submodule("decoder.layers.0.self_attn.q_proj")
    torch.testing.assert_close(m(x), F.linear(x, base["decoder.layers.0.self_attn.q_proj"], m.bias))


def test_matches_upstream_lora_linear(tmp_path):
    lora = pytest.importorskip("voxcpm.modules.layers.lora")
    t = write_adapter(tmp_path / "a", r=4, alpha=8)
    lin = nn.Linear(12, 16)
    up = lora.LoRALinear(nn.Linear(12, 16), r=4, alpha=8)
    up.weight.data.copy_(lin.weight)
    up.bias.data.copy_(lin.bias)
    up.lora_A.data.copy_(t[f"{ATTN}.q_proj.lora_A"])
    up.lora_B.data.copy_(t[f"{ATTN}.q_proj.lora_B"])
    ours = A.AdapterLinear(lin)
    ours.lora_A, ours.lora_B, ours.scaling = t[f"{ATTN}.q_proj.lora_A"], t[f"{ATTN}.q_proj.lora_B"], 2.0
    x = torch.randn(4, 12)
    torch.testing.assert_close(ours(x), up(x))


def test_switching_adapters_copy_and_resident(tmp_path):
    for k in range(2):
        write_adapter(tmp_path / f"a{k}", seed=k)
    dirs = {f"a{k}": str(tmp_path / f"a{k}") for k in range(2)}
    root = toy_attn()
    A.inject_dit(root, dirs)
    x = torch.randn(2, 12)
    outs = {}
    for resident in (True, False):
        bank = A.AdapterBank(dirs, prefix="feat_decoder.estimator.", dtype=torch.float32, resident=resident)
        q = root.get_submodule("decoder.layers.0.self_attn.q_proj")
        for name in ("a0", "a1", None, "a0"):
            bank.use(root, name)
            outs.setdefault((resident, name), q(x).detach())
            torch.testing.assert_close(q(x), outs[(resident, name)])
        bank.use(root, None)
        bank._current.clear()
        bank._layers.clear()
    for name in ("a0", "a1", None):
        torch.testing.assert_close(outs[(True, name)], outs[(False, name)])
    assert not torch.allclose(outs[(True, "a0")], outs[(True, "a1")])
    with pytest.raises(KeyError):
        bank.use(root, "nope")


def test_bank_refuses_a_mismatched_model(tmp_path):
    write_adapter(tmp_path / "a")
    root = toy_attn()
    A.inject(root, ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"])  # gate_proj has no LoRA in the adapter
    bank = A.AdapterBank({"a": str(tmp_path / "a")}, prefix="feat_decoder.estimator.", dtype=torch.float32)
    with pytest.raises(ValueError, match="missing"):
        bank.use(root, "a")


@pytest.mark.skipif(not ADAPTERS_MODEL, reason="the adapters model is not downloaded")
def test_real_adapters_cover_the_locdit_layout():
    """The real adapters' LocDiT tensors find their layers in the LocDiT's module tree (names and shapes as in
    voxcpm's local_dit_v2: 12 layers, hidden 1024, q 2048, k / v 256), with s = alpha / r = 2."""
    model = ADAPTERS_MODEL[0]
    found = A.find(model)
    assert set(found) >= set(CURRENT_ADAPTER.values()) and len(found) == 11
    cfg = A.shared_config(found)
    assert cfg["enable_dit"] and cfg["enable_lm"] and not cfg["enable_proj"]
    assert sorted(cfg["target_modules_dit"]) == ["k_proj", "o_proj", "q_proj", "v_proj"]
    assert A.scaling(A.read_config(found["saudi"])) == 2.0
    root = nn.Module()
    root.decoder = nn.Module()
    root.decoder.layers = nn.ModuleList()
    for _ in range(12):
        layer = nn.Module()
        layer.self_attn = nn.Module()
        for n, (i, o) in {"q_proj": (1024, 2048), "k_proj": (1024, 256), "v_proj": (1024, 256),
                          "o_proj": (2048, 1024)}.items():
            layer.self_attn.add_module(n, nn.Linear(i, o, bias=False, device="meta"))
        layer.mlp = nn.Module()
        layer.mlp.gate_proj = nn.Linear(1024, 4096, bias=False, device="meta")
        root.decoder.layers.append(layer)
    assert A.inject_dit(root, found) == 48
    bank = A.AdapterBank({"saudi": found["saudi"]}, prefix="feat_decoder.estimator.", dtype=torch.float32)
    bank.use(root, "saudi")
    for n, m in bank.layers(root).items():
        assert m.lora_A.shape == (64, m.in_features) and m.lora_B.shape == (m.out_features, 64), n


def test_routing_follows_the_evaluation_baseline():
    # every evaluation group's dialect goes through the group's baseline adapter, the others through the base model
    for g, d in DIALECT_OF_GROUP.items():
        assert A.adapter_of(d) == CURRENT_ADAPTER.get(g), g
    assert A.adapter_of("msa") is None and A.adapter_of("moroccan") is None
    assert A.adapter_of("lebanese") == A.adapter_of("syrian") == "jordanian"
    assert A.adapter_of("gulf") == "emirati" and A.adapter_of("bahraini") == "qatari"
    assert set(A.ADAPTER_OF_DIALECT) <= set(dialects.DIALECTS)
    assert A.adapter_of("egyptian", {"egyptian": None}) is None  # --adapter-map


class FakeSet:
    def __init__(self, n):
        self.n = n

    def length(self, i):
        return 10 + (i * 7) % 30


def test_plan_batches_hold_one_adapter_each():
    rows = [{"dialect": d} for d in ["egyptian"] * 50 + ["msa"] * 10 + ["lebanese"] * 5 + ["syrian"] * 20]
    adapters = [A.adapter_of(r["dialect"]) for r in rows]
    datasets = {}
    for a in set(adapters):
        ids = [g for g, x in enumerate(adapters) if x == a]
        datasets[a] = (FakeSet(len(ids)), ids)
    plan = T.make_plan(rows, adapters, datasets, 400, 120, "dialect", seed=0)
    assert plan == T.make_plan(rows, adapters, datasets, 400, 120, "dialect", seed=0)  # deterministic
    drawn = [g for a, b in plan for g, _ in b]
    assert len(drawn) == 400
    for a, b in plan:
        assert all(adapters[g] == a for g, _ in b)
        assert max(datasets[a][0].length(datasets[a][1].index(g)) for g, _ in b) * len(b) <= 120 or len(b) == 1
    per = {d: sum(rows[g]["dialect"] == d for g in drawn) for d in ("egyptian", "msa", "lebanese", "syrian")}
    assert all(60 <= v <= 140 for v in per.values()), per  # dialects evenly (100 each)
    reps = [rep for _, b in plan for g, rep in b if rows[g]["dialect"] == "lebanese"]
    assert max(reps) > 0  # 5 clips drawn ~100 times: other draws of the same clip
    w = T.weights(rows, adapters, "adapter")
    tot = {}
    for wi, a in zip(w, adapters):
        tot[a] = tot.get(a, 0) + wi
    assert max(tot.values()) - min(tot.values()) < 1e-9  # each adapter (and the base model) equally likely

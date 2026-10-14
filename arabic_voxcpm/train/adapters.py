"""The current model's dialect adapters (VoxCPM-Kernel/VoxCPM2-Arabic-Dialects) on the PyTorch model.

The adapters are LoRAs over a plain VoxCPM2 (``adapters/<name>/{lora_config.json, lora_weights.safetensors}``): r = 64,
alpha = 128 on the q/k/v/o projections of base_lm, residual_lm and the LocDiT (feat_decoder.estimator). A layer's
output is W x + b + s * B (A x) with s = alpha / r, what upstream voxcpm's LoRALinear and the inference engine
(nanovllm_voxcpm models/voxcpm2/lora_loader.py: scaling = alpha / rank, folded into B) compute; merged, the weight is
W + s * B @ A (``delta``, tests/test_adapters.py checks it against the engine's loader).

Here the targeted nn.Linear layers become ``AdapterLinear``: the same weight and bias (the same state_dict keys, so a
LocDiT checkpoint is still the engine's), plus one adapter's A, B and s held as plain attributes, outside the
parameters and the state_dict. ``AdapterBank.use(root, name)`` switches every such layer of a module tree to one
adapter (None: the base model) between batches: by reference when the bank sits on the model's device (the LocDiT's
LoRAs, ~13 MB each), by copy into one set of buffers otherwise (the whole model's, ~72 MB each in bf16).

Routing: a clip goes through the adapter of its dialect, the one the evaluation's baseline uses for that group
(eval.build.CURRENT_ADAPTER over DIALECT_OF_GROUP); dialects without one use the base model.
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from arabic_voxcpm.eval.build import CURRENT_ADAPTER, DIALECT_OF_GROUP

ADAPTER_OF_DIALECT: dict[str, str] = {DIALECT_OF_GROUP[g]: a for g, a in CURRENT_ADAPTER.items()}
WEIGHTS, CONFIG = "lora_weights.safetensors", "lora_config.json"


def adapter_of(dialect: str, mapping: dict | None = None) -> str | None:
    """The adapter a clip of ``dialect`` (train.dialects canonical name) goes through; None = the base model."""
    return (ADAPTER_OF_DIALECT if mapping is None else mapping).get(dialect)


def find(model_dir: str) -> dict[str, str]:
    """Adapter name -> directory: the subdirectories of <model_dir>/adapters with LoRA weights (as the engine's
    dialect_tts.find_adapters)."""
    root = os.path.join(model_dir, "adapters")
    if not os.path.isdir(root):
        return {}
    return {n: os.path.join(root, n) for n in sorted(os.listdir(root))
            if os.path.isfile(os.path.join(root, n, WEIGHTS))}


def read_config(adapter_dir: str) -> dict:
    cfg = json.load(open(os.path.join(adapter_dir, CONFIG), encoding="utf-8"))
    return cfg.get("lora_config", cfg)


def scaling(cfg: dict) -> float:
    """s = alpha / r (upstream LoRALinear._base_scaling, the engine's _build_module_payload)."""
    return float(cfg.get("alpha", cfg.get("lora_alpha"))) / int(cfg.get("r", cfg.get("rank")))


def delta(a: torch.Tensor, b: torch.Tensor, s: float) -> torch.Tensor:
    """The LoRA's weight delta: merged weight = W + s * B @ A (A [r, in], B [out, r])."""
    return s * (b.float() @ a.float())


def shared_config(adapters: dict[str, str]) -> dict:
    """The adapters' common targets (the engine needs one LoRA layout for all of them)."""
    keys = ("enable_lm", "enable_dit", "enable_proj", "target_modules_lm", "target_modules_dit", "target_proj_modules")
    shared = None
    for name, path in adapters.items():
        cfg = read_config(path)
        t = {k: sorted(cfg[k]) if isinstance(cfg.get(k), list) else cfg.get(k) for k in keys}
        if shared is None:
            shared = t
        elif t != shared:
            raise ValueError(f"adapter {name!r} targets other modules than the others: {t} vs {shared}")
    return shared or {}


class AdapterLinear(nn.Module):
    """nn.Linear plus one switchable, frozen LoRA: y = W x + b + s * B (A x). lora_A / lora_B are plain tensor
    attributes (or None: the base layer), never parameters or state."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.in_features, self.out_features = base.in_features, base.out_features
        self.weight = base.weight
        self.bias = base.bias
        self.lora_A: torch.Tensor | None = None
        self.lora_B: torch.Tensor | None = None
        self.scaling = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self.lora_A is None or self.scaling == 0.0:
            return y
        a, b = self.lora_A, self.lora_B
        if a.dtype != x.dtype and not torch.is_autocast_enabled(x.device.type):
            a, b = a.to(x.dtype), b.to(x.dtype)
        return y + F.linear(F.linear(x, a), b) * self.scaling

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


def inject(root: nn.Module, targets: list[str]) -> list[str]:
    """Every nn.Linear of ``root`` named one of ``targets`` (the last name component, as upstream's
    apply_lora_to_named_linear_modules) becomes an AdapterLinear. Returns their names (relative to root)."""
    done = []
    for name, module in list(root.named_modules()):
        if type(module) is nn.Linear and name.split(".")[-1] in targets:
            parent = root.get_submodule(name.rsplit(".", 1)[0]) if "." in name else root
            setattr(parent, name.split(".")[-1], AdapterLinear(module))
            done.append(name)
    return done


def inject_model(model: nn.Module, cfg: dict) -> int:
    """The adapters' layout on a VoxCPM2 model (upstream VoxCPM2Model._apply_lora): enable_lm -> base_lm and
    residual_lm, enable_dit -> feat_decoder.estimator, enable_proj -> the projections on the model itself."""
    n = 0
    if cfg.get("enable_lm"):
        for lm in (model.base_lm, model.residual_lm):
            n += len(inject(lm, cfg["target_modules_lm"]))
    if cfg.get("enable_dit"):
        n += len(inject(model.feat_decoder.estimator, cfg["target_modules_dit"]))
    if cfg.get("enable_proj"):
        for attr in cfg.get("target_proj_modules", []):
            if type(getattr(model, attr, None)) is nn.Linear:
                setattr(model, attr, AdapterLinear(getattr(model, attr)))
                n += 1
    return n


def inject_dit(estimator: nn.Module, adapters: dict[str, str]) -> int:
    """The adapters' LocDiT layout on a bare estimator (feat_decoder.estimator)."""
    cfg = shared_config(adapters)
    return len(inject(estimator, cfg["target_modules_dit"])) if cfg.get("enable_dit") else 0


class AdapterBank:
    """Every adapter's LoRA tensors, switchable onto a module tree's AdapterLinear layers.

    adapters: name -> directory, or name -> {module name: (A, B)} with ``alpha`` / ``r`` given (tests).
    prefix: only tensors under this module path (e.g. "feat_decoder.estimator." for a bare LocDiT), stripped.
    resident: the tensors stay on ``device`` and layers point at them; otherwise they stay on the CPU (pinned) and
    are copied into one set of buffers per layer when the adapter changes."""

    def __init__(self, adapters: dict, prefix: str = "", device="cpu", dtype=torch.bfloat16, resident: bool = True,
                 alpha: float | None = None, r: int | None = None):
        from safetensors import safe_open

        self.prefix, self.device, self.dtype, self.resident = prefix, torch.device(device), dtype, resident
        self.tensors: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
        self.scale: dict[str, float] = {}
        for name, src in adapters.items():
            if isinstance(src, dict):
                mods, s = src, alpha / r
            else:
                cfg = read_config(src)
                s = scaling(cfg)
                raw = {}
                with safe_open(os.path.join(src, WEIGHTS), framework="pt", device="cpu") as f:
                    for k in f.keys():
                        if k.startswith(prefix) and k.endswith((".lora_A", ".lora_B")):
                            raw[k[len(prefix):]] = f.get_tensor(k)
                mods = {}
                for k in raw:
                    if k.endswith(".lora_A"):
                        m = k[: -len(".lora_A")]
                        mods[m] = (raw[k], raw[m + ".lora_B"])
                ranks = {a.shape[0] for a, _ in mods.values()}
                assert ranks <= {int(cfg.get("r", cfg.get("rank")))}, f"{name}: ranks {ranks} vs config"
            store = self.device if resident else torch.device("cpu")
            pin = not resident and torch.cuda.is_available()
            self.tensors[name] = {m: tuple(t.to(store, dtype).contiguous().pin_memory() if pin
                                           else t.to(store, dtype).contiguous() for t in ab) for m, ab in mods.items()}
            self.scale[name] = s
        self.names = sorted(self.tensors)
        self._layers: dict[int, dict[str, AdapterLinear]] = {}
        self._current: dict[int, str | None] = {}
        self._buffers: dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}

    def layers(self, root: nn.Module) -> dict[str, AdapterLinear]:
        key = id(root)
        if key not in self._layers:
            self._layers[key] = {n: m for n, m in root.named_modules() if isinstance(m, AdapterLinear)}
            self._current[key] = "<unset>"
        return self._layers[key]

    def use(self, root: nn.Module, name: str | None, strict: bool = True) -> None:
        """Switch root's AdapterLinear layers to adapter ``name`` (None: no LoRA). strict: every adapter tensor of
        the prefix must find its layer (off for a student with fewer LocDiT layers: the dropped layers' LoRAs go
        unused); every layer must get one either way."""
        layers = self.layers(root)
        key = id(root)
        if self._current[key] == name:
            return
        if name is None:
            for m in layers.values():
                m.scaling = 0.0
            self._current[key] = None
            return
        if name not in self.tensors:
            raise KeyError(f"unknown adapter {name!r}; adapters: {', '.join(self.names)}")
        mods = self.tensors[name]
        missing = sorted(set(layers) - set(mods))
        extra = sorted(set(mods) - set(layers))
        if missing or (strict and extra):
            raise ValueError(f"adapter {name!r} does not match the model's LoRA layers: missing {missing[:3]}, "
                             f"unused {extra[:3]} ({len(missing)} / {len(extra)})")
        s = self.scale[name]
        for n, m in layers.items():
            a, b = mods[n]
            if self.resident:
                m.lora_A, m.lora_B = a, b
            else:
                buf = self._buffers.setdefault(key, {})
                if n not in buf:
                    dev = m.weight.device
                    buf[n] = (torch.empty(a.shape, device=dev, dtype=self.dtype),
                              torch.empty(b.shape, device=dev, dtype=self.dtype))
                buf[n][0].copy_(a, non_blocking=True)
                buf[n][1].copy_(b, non_blocking=True)
                m.lora_A, m.lora_B = buf[n]
            m.scaling = s
        self._current[key] = name


def load_teacher(model_dir: str, device="cuda", adapters: dict[str, str] | None = None):
    """The base VoxCPM2 of ``model_dir`` (bf16, eval, no AudioVAE) with its adapters' layout injected, and the bank
    of all its adapters (on the CPU, copied in when the adapter changes). bank.use(model, name) before each batch."""
    from voxcpm.model.voxcpm2 import VoxCPM2Model

    adapters = find(model_dir) if adapters is None else adapters
    assert adapters, f"no adapters under {model_dir}/adapters"
    model = VoxCPM2Model.from_local(model_dir, optimize=False, device=str(device))  # bf16 on the device
    model.audio_vae = None  # not needed for teacher-forced LMs and the LocDiT
    n = inject_model(model, shared_config(adapters))
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    for p in model.parameters():
        p.requires_grad = False
    bank = AdapterBank(adapters, device=device, dtype=torch.bfloat16, resident=False)
    print(f"adapter teacher: {n} LoRA layers, adapters {', '.join(bank.names)}", flush=True)
    return model, bank

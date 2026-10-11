"""The upstream VoxCPM2 model in training form: frozen bf16 base, fp32 LoRA, trainable dialect tag embeddings,
activation checkpointing per transformer layer; saving the trainable state; merging it into a plain model directory.
"""

from __future__ import annotations

import functools
import json
import os
import shutil

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from arabic_voxcpm.train import dialects

LORA_DEFAULT = dict(enable_lm=True, enable_dit=True, enable_proj=True, r=64, alpha=64, dropout=0.0,
                    target_modules_lm=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    target_modules_dit=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])


class TaggedEmbedding(nn.Module):
    """The base embedding, with the dialect tags' rows taken from a small trainable table."""

    def __init__(self, base: nn.Embedding, tag_ids: list[int], init: torch.Tensor):
        super().__init__()
        self.base = base
        self.register_buffer("tag_ids", torch.tensor(tag_ids), persistent=False)
        self.first = min(tag_ids)
        assert tag_ids == list(range(self.first, self.first + len(tag_ids)))
        self.tag_embed = nn.Parameter(init.float().clone())

    @property
    def weight(self):
        return self.base.weight

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        out = self.base(ids)
        k = ids - self.first
        is_tag = (k >= 0) & (k < len(self.tag_ids))
        if is_tag.any():
            out = torch.where(is_tag.unsqueeze(-1), self.tag_embed[k.clamp(0, len(self.tag_ids) - 1)].to(out.dtype), out)
        return out


class Tokenizer:
    """Text -> ids exactly as the model tokenizes at inference (upstream wrapper), plus the dialect tag ids."""

    def __init__(self, model_dir: str):
        from transformers import LlamaTokenizerFast
        from voxcpm.model.utils import mask_multichar_chinese_tokens

        self.raw = LlamaTokenizerFast.from_pretrained(model_dir)
        self.wrapped = mask_multichar_chinese_tokens(self.raw)

    def __call__(self, text: str) -> list[int]:
        return list(self.wrapped(text))

    def tag_id(self, dialect: str) -> int:
        return self.raw.convert_tokens_to_ids(dialects.tag_token(dialect))


def _checkpointed(layer: nn.Module):
    fwd = layer.forward

    @functools.wraps(fwd)
    def run(*args, **kwargs):
        if torch.is_grad_enabled():
            return checkpoint(fwd, *args, use_reentrant=False, **kwargs)
        return fwd(*args, **kwargs)

    layer.forward = run


def load(model_dir: str, lora: dict | None = None, device: str = "cuda"):
    from voxcpm.model.voxcpm2 import LoRAConfig, VoxCPM2Model
    from voxcpm.modules.minicpm4.model import MiniCPMDecoderLayer

    cfg = dict(LORA_DEFAULT, **(lora or {}))
    model = VoxCPM2Model.from_local(model_dir, training=True, lora_config=LoRAConfig(**cfg))
    tok = Tokenizer(model_dir)
    tag_ids = [tok.tag_id(d) for d in dialects.DIALECTS]
    emb = model.base_lm.embed_tokens
    # each tag starts from the mean embedding of its English name's tokens
    init = torch.stack([emb.weight[torch.tensor(tok(d))].float().mean(0) for d in dialects.DIALECTS])
    model.base_lm.embed_tokens = TaggedEmbedding(emb, tag_ids, init)
    for name, p in model.named_parameters():
        p.requires_grad = "lora_" in name or "tag_embed" in name
        if not p.requires_grad and p.dtype == torch.float32 and "audio_vae" not in name:
            p.data = p.data.to(torch.bfloat16)
    for m in model.modules():
        if isinstance(m, MiniCPMDecoderLayer):
            _checkpointed(m)
    model.lora_cfg = cfg
    return model.to(device), tok


def trainable_state(model) -> dict[str, torch.Tensor]:
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}


def save(model, path: str, extra: dict | None = None) -> None:
    from safetensors.torch import save_file

    os.makedirs(path, exist_ok=True)
    save_file(trainable_state(model), os.path.join(path, "trainable.safetensors"))
    json.dump({"lora": model.lora_cfg, "dialects": dialects.DIALECTS, **(extra or {})},
              open(os.path.join(path, "train_state.json"), "w"), indent=1)


def load_trainable(model, path: str) -> None:
    from safetensors.torch import load_file

    state = load_file(os.path.join(path, "trainable.safetensors"))
    missing = set(n for n, p in model.named_parameters() if p.requires_grad) - set(state)
    assert not missing, f"missing trainable tensors: {sorted(missing)[:5]}"
    model.load_state_dict(state, strict=False)


def export_merged(src_dir: str, ckpt: str, out_dir: str) -> None:
    """A plain VoxCPM2 model directory: every LoRA merged into its weight (W + s * B @ A), the tag rows written into
    the embedding, the tags registered as special tokens (so "<reserved_N>" in text is one id), dialects.json."""
    from safetensors.torch import load_file, save_file

    state = load_file(os.path.join(src_dir, "model.safetensors"))
    train = load_file(os.path.join(ckpt, "trainable.safetensors"))
    cfg = json.load(open(os.path.join(ckpt, "train_state.json")))["lora"]
    scale = cfg["alpha"] / cfg["r"]
    merged = 0
    for name, a in train.items():
        if not name.endswith(".lora_A"):
            continue
        base = name[: -len(".lora_A")]
        b = train[base + ".lora_B"]
        w = state[base + ".weight"]
        state[base + ".weight"] = (w.float() + scale * (b.float() @ a.float())).to(w.dtype)
        merged += 1
    tags = train["base_lm.embed_tokens.tag_embed"]
    emb = state["base_lm.embed_tokens.weight"]
    tok = Tokenizer(src_dir)
    for d, row in zip(dialects.DIALECTS, tags):
        emb[tok.tag_id(d)] = row.to(emb.dtype)
    os.makedirs(out_dir, exist_ok=True)
    save_file(state, os.path.join(out_dir, "model.safetensors"))
    for f in os.listdir(src_dir):
        if f not in ("model.safetensors", "adapters", "adapters.json", "README.md") and os.path.isfile(os.path.join(src_dir, f)):
            shutil.copy(os.path.join(src_dir, f), out_dir)
    tok.raw.add_special_tokens({"additional_special_tokens": [dialects.tag_token(d) for d in dialects.DIALECTS]},
                               replace_additional_special_tokens=False)
    tok.raw.save_pretrained(out_dir)
    json.dump({"tags": {d: dialects.tag_token(d) for d in dialects.DIALECTS}, "aliases": dialects.ALIASES},
              open(os.path.join(out_dir, "dialects.json"), "w"), indent=1)
    print(f"merged {merged} LoRA layers and {len(tags)} dialect tags -> {out_dir}")

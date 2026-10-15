"""teacher-v3.1: the continuation from teacher-v3 (load_init) and its corpus weights."""

import json
import os

import pytest
import torch
import torch.nn as nn

from arabic_voxcpm.train import dataset as D
from arabic_voxcpm.train import dialects
from arabic_voxcpm.train import model as M

V3 = "sada=0.5,synth:hejazi=2,synth:najdi=2,synth:gulf=2,synth:saudi=2,synth:msa=2"
V31 = ("sada=0.5,sada:saudi=2,sada:egyptian=1,lahgtna:bahraini=2,lahgtna:egyptian=2,"
       "synth:hejazi=2,synth:najdi=2,synth:gulf=2,synth:saudi=1,synth:msa=2")


def test_v31_weights():
    rows = [{"source": s, "dialect": d} for s, d in [
        ("lahgtna", "bahraini"), ("lahgtna", "egyptian"), ("sada", "egyptian"), ("sada", "saudi"), ("sada", "najdi"),
        ("synth", "saudi"), ("synth", "najdi"), ("lahgtna", "sudanese"), ("synth", "iraqi")]]
    assert D.row_weights(rows, D.parse_weights(V3)) == [1, 1, 0.5, 0.5, 0.5, 2, 2, 1, 1]
    # Bahraini x2, Egyptian x2 (SADA's few Egyptian rows too), SADA Saudi 0.5 -> 2, Saudi synthetic 2 -> 1; the rest as v3
    assert D.row_weights(rows, D.parse_weights(V31)) == [2, 2, 1, 2, 0.5, 1, 2, 1, 1]


class Toy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(4, 3))
        self.tag_embed = nn.Parameter(torch.randn(2, 3))
        self.frozen = nn.Parameter(torch.randn(3), requires_grad=False)
        self.lora_cfg = cfg


def test_load_init(tmp_path):
    cfg = dict(M.LORA_DEFAULT, enable_dit=False, enable_proj=False, train_modules=["base_lm"])
    src = Toy(cfg)
    M.save(src, str(tmp_path / "best"), {"step": 1500})
    dst = Toy(dict(cfg))
    frozen = dst.frozen.detach().clone()
    st = M.load_init(dst, str(tmp_path / "best"))
    assert st["step"] == 1500
    assert torch.equal(dst.lora_A, src.lora_A) and torch.equal(dst.tag_embed, src.tag_embed)
    assert torch.equal(dst.frozen, frozen)  # only the trainable tensors come from the checkpoint


def test_load_init_refuses_other_lora_or_dialects(tmp_path):
    cfg = dict(M.LORA_DEFAULT, train_modules=["base_lm"])
    M.save(Toy(cfg), str(tmp_path / "best"))
    with pytest.raises(ValueError, match="LoRA"):
        M.load_init(Toy(dict(cfg, r=32)), str(tmp_path / "best"))
    p = os.path.join(tmp_path, "best", "train_state.json")
    st = json.load(open(p))
    json.dump(dict(st, dialects=list(reversed(dialects.DIALECTS))), open(p, "w"))
    with pytest.raises(ValueError, match="dialects"):
        M.load_init(Toy(cfg), str(tmp_path / "best"))

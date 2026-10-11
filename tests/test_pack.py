"""Our packing matches the upstream packer on the same latents."""

import torch

from arabic_voxcpm.train import pack


def _upstream():
    # the packer module alone: voxcpm.training's __init__ pulls in datasets / argbind
    import importlib.util
    import os

    import voxcpm

    path = os.path.join(os.path.dirname(voxcpm.__file__), "training", "packers.py")
    spec = importlib.util.spec_from_file_location("upstream_packers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    p = mod.AudioFeatureProcessingPacker.__new__(mod.AudioFeatureProcessingPacker)
    p.audio_start_id, p.audio_end_id, p.audio_prompt_start_id, p.audio_prompt_end_id = 101, 102, 103, 104
    p.patch_size, p.feat_dim = 4, 64
    return p


def test_plain_and_ref_match_upstream():
    torch.manual_seed(0)
    up = _upstream()
    text = torch.tensor([5, 6, 7], dtype=torch.int32)
    lat = {"tgt": torch.randn(37, 64), "ref": torch.randn(22, 64)}

    def feats(which):
        return pack.patches(lat[which]).unsqueeze(0), lat[which].shape[0] / 25

    up.extract_audio_feats = lambda tok: feats("ref" if tok.numel() == 1 and tok.item() == 1 else "tgt")
    got = up.process_tts_data(torch.tensor([0]), text, is_prompt=False)
    ours = pack.plain(text.tolist(), pack.patches(lat["tgt"]))
    keys = ["text_tokens", "audio_feats", "text_mask", "audio_mask", "loss_mask", "labels"]
    for k, v in zip(keys, got[:6]):
        assert torch.equal(ours[k].to(v.dtype), v), k
    got = up.process_tts_data_with_ref(torch.tensor([1]), torch.tensor([0]), text)
    ours = pack.with_ref(pack.patches(lat["ref"]), text.tolist(), pack.patches(lat["tgt"]))
    for k, v in zip(keys, got[:6]):
        assert torch.equal(ours[k].to(v.dtype), v), k


def test_collate_pads_right():
    a = pack.plain([1, 2], torch.randn(3, 4, 64))
    b = pack.plain([1, 2, 3, 4], torch.randn(5, 4, 64))
    batch = pack.collate([a, b])
    assert batch["text_tokens"].shape == (2, 11) and batch["audio_feats"].shape == (2, 11, 4, 64)
    assert batch["loss_mask"][0, 6:].sum() == 0 and batch["labels"][0].tolist().index(1) == 5


def test_tags_tokenize_like_inference():
    """A tagged text is the tag's one id followed by the text's ids without the leading word-start mark."""
    import os

    import pytest

    from arabic_voxcpm.train import dialects
    from arabic_voxcpm.train.model import Tokenizer

    model = "/workspace/models/VoxCPM2-Arabic-Dialects"
    if not os.path.isdir(model):
        pytest.skip("needs the model's tokenizer")
    tok = Tokenizer(model)
    for d in ("najdi", "msa", "moroccan"):
        ids = tok(dialects.tag_token(d) + "هلا والله")
        assert ids[0] == tok.tag_id(d) and ids[1:] == tok("هلا والله")[1:]

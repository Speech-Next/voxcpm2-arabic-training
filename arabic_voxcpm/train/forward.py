"""VoxCPM2's training forward (upstream VoxCPM2Model.forward) without the work the loss never sees.

Upstream runs the local encoder on every position (text and padding included) and the LocDiT flow-matching loss on
every position, masked afterwards; a training batch is about a third target audio, so most of the LocDiT's work is
discarded. Here the encoder runs on audio positions only and the LocDiT on loss positions only. Both are per-position
computations and the loss weights are the mask itself (adaptive weighting with p = 0), so the loss is the same
expectation; only the random draws (t, noise, CFG dropout) are taken for fewer rows. Measured on a 2754-position
batch: LocDiT loss 1.9 -> 0.53 s forward + backward. tests/test_forward.py checks the equivalence.
"""

from __future__ import annotations

import torch


def forward(model, text_tokens, text_mask, audio_feats, audio_mask, loss_mask, labels, progress: float = 0.0, **_):
    dtype = model._dtype()
    dev = model.device
    text_tokens = text_tokens.to(dev, dtype=torch.long)
    text_mask = text_mask.to(dev, dtype=dtype)
    audio_feats = audio_feats.to(dev, dtype=dtype)
    am = audio_mask.to(dev).bool()
    lm_mask = loss_mask.to(dev).bool()
    labels = labels.to(dev, dtype=torch.long)
    B, T, P, D = audio_feats.shape

    enc = model.enc_to_lm_proj(model.feat_encoder(audio_feats[am].unsqueeze(0)))[0]  # [N_audio, H]
    feat_embed = torch.zeros(B, T, enc.shape[-1], device=dev, dtype=enc.dtype)
    feat_embed[am] = enc

    scale_emb = getattr(model.config.lm_config, "scale_emb", 1.0)
    if not getattr(model.config.lm_config, "use_mup", False):
        scale_emb = 1.0
    text_embed = model.base_lm.embed_tokens(text_tokens) * scale_emb
    amf = am.to(dtype).unsqueeze(-1)
    combined = text_mask.unsqueeze(-1) * text_embed + amf * feat_embed

    enc_out, _ = model.base_lm(inputs_embeds=combined, is_causal=True)
    enc_out = enc_out.to(dtype)
    enc_out = model.fsq_layer(enc_out) * amf + enc_out * text_mask.unsqueeze(-1)
    lm_hidden = torch.cat((torch.zeros_like(enc_out[:, :1]), enc_out[:, :-1]), dim=1)

    res_in = model.fusion_concat_proj(torch.cat((enc_out, amf * feat_embed), dim=-1))
    res_out, _ = model.residual_lm(inputs_embeds=res_in, is_causal=True)
    res_out = res_out.to(dtype)
    res_hidden = torch.cat((torch.zeros_like(res_out[:, :1]), res_out[:, :-1]), dim=1)

    # LocDiT on the loss positions only
    dit_hidden = torch.cat((model.lm_to_dit_proj(lm_hidden[lm_mask]), model.res_to_dit_proj(res_hidden[lm_mask])), dim=-1)
    feat_cond = torch.cat((torch.zeros_like(audio_feats[:, :1]), audio_feats[:, :-1]), dim=1)
    gt = audio_feats[lm_mask].transpose(1, 2).contiguous()  # [M, D, P]
    cond = feat_cond[lm_mask].transpose(1, 2).contiguous()
    tgt_mask = torch.ones(gt.shape[0], 1, P, device=dev, dtype=dtype)
    diff_loss = model.feat_decoder.compute_loss(gt, dit_hidden, cond=cond, tgt_mask=tgt_mask, progress=progress)

    stop_logits = model.stop_head(model.stop_actn(model.stop_proj(lm_hidden)))
    stop_losses = model.stop_loss(stop_logits.transpose(1, 2), labels)
    lmf = lm_mask.to(dtype)
    stop_loss = (stop_losses * lmf).sum() / torch.clamp(lmf.sum(), min=1.0)
    return {"loss/diff": diff_loss, "loss/stop": stop_loss}

# VoxCPM2 Arabic training

Data, evaluation, training and distillation for the Arabic dialect VoxCPM2 model. The inference library lives in
[`Speech-Next/voxcpm2-arabic-dialects-turbo`](https://github.com/Speech-Next/voxcpm2-arabic-dialects-turbo); the
plan and its issues are there too (roadmap: issue #1).

| directory | what | issue |
|---|---|---|
| `data/` | Arabic text and speech: download, segmentation, filters, AudioVAE latents, manifests | #3 |
| `eval/` | held-out Arabic evaluation: texts, voices, metrics, reports | #2 |
| `train/` | LoRA fine-tuning of the teacher on real Arabic speech | #5 |
| `distill/` | teacher data, LocDiT and LM distillation | #6, #7, #8 |

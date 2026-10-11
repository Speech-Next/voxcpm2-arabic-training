# VoxCPM2 Arabic training

Data, evaluation, training and distillation for the Arabic VoxCPM2 model. The inference library lives in
[`Speech-Next/voxcpm2-arabic-dialects-turbo`](https://github.com/Speech-Next/voxcpm2-arabic-dialects-turbo), and so
does the plan (roadmap: issue #1).

**The model:** one VoxCPM2 for all Arabic dialects. The dialect is chosen by a tag token in front of the text (one
unused `<reserved_N>` token per dialect), and a LoRA over the LMs, the LocDiT and the projections is merged into the
weights. There are no per-dialect adapters.

17 dialects:

| region | dialects |
|---|---|
| — | msa |
| Egypt | egyptian |
| Levant | palestinian, lebanese, syrian |
| Iraq | iraqi |
| Gulf | saudi, najdi, hejazi, gulf, bahraini |
| South | yemeni, sudanese |
| Maghreb | libyan, tunisian, algerian, moroccan |

Older names stay usable as aliases: emirati / qatari → gulf, jordanian → palestinian.

## Layout

| package | what | issue |
|---|---|---|
| `arabic_voxcpm.data` | corpora → AudioVAE latent shards + manifests (`lahgtna`, `sada`), one source shard at a time | #3 |
| `arabic_voxcpm.eval` | held-out evaluation set, generation through the inference library, scoring, paired reports | #2 |
| `arabic_voxcpm.train` | teacher training (tags, packing, dataset, model, forward, loop), merged export | #5 |
| `scripts/run_teacher.sh` | the whole teacher pipeline, unattended | |

## Data

| corpus | used | dialects | license |
|---|---|---|---|
| [`Rabe3/lahgtna-arabic-tts-24khz`](https://huggingface.co/datasets/Rabe3/lahgtna-arabic-tts-24khz) | ~480 h after filters, 688 speakers | 13 | none declared |
| [SADA](https://huggingface.co/datasets/MohamedRashad/SADA22) (SDAIA, Saudi Broadcasting Authority) | ~51 h, single speaker, clean environment | Najdi, Hejazi, Gulf, MSA, ... | CC BY-NC-SA 4.0 |

Latents are fp16 at 25 Hz x 64, about 3.2 KB per second of audio, all held in memory during training.

```bash
python -m arabic_voxcpm.data.lahgtna       # -> /workspace/data/arabic/lahgtna/{latents,manifest,eval_audio}
python -m arabic_voxcpm.data.sada          # -> /workspace/data/arabic/sada/...
```

## Evaluation

The set is held out from training: speakers, clip ids and texts are listed in `heldout.json`. It has 30 texts per
group (17 dialect groups), and reference voices come from held-out speakers.

Metrics:
- Whisper large-v3 CER / WER, after Arabic normalization;
- UTMOS and DNSMOS;
- WavLM-SV similarity to the cloned voice;
- the ASR's error on the real recordings, the floor any model is compared against.

Reports compare per group, with paired bootstrap 95 % intervals against a baseline run.

```bash
python -m arabic_voxcpm.eval.build --out /workspace/data/arabic/eval/v1
PYTHONPATH=.:<inference repo> <inference python> -m arabic_voxcpm.eval.generate --out runs/<name> --model <model dir> \
    [--adapter-map /workspace/data/arabic/eval/v1/dialect_map.json]      # tagged models
python -m arabic_voxcpm.eval.score runs/<name> [--gt /workspace/data/arabic/eval/v1]
python -m arabic_voxcpm.eval.report --base runs/<baseline> runs/<name> --md report.md
```

Generation runs in bf16 by default. A new model's SmoothQuant statistics are not calibrated, so int8 would measure the
calibration, not the model.

## Training

```bash
scripts/run_teacher.sh teacher-v1 4200 --val-every 500 --save-every 500
```

The script waits for the corpora, builds the evaluation set, measures the baseline (the current adapters), trains,
exports `best` and `last` as merged model directories, evaluates them and writes `report.md`.

**Loss and model:**
- VoxCPM2's own loss: LocDiT flow matching plus stop, 1:1.
- The base model is frozen in bf16; the LoRA (r 64, on q/k/v/o and the FFN) and the tag embeddings are fp32: 105.6M
  trainable parameters.

**Samples:**
- 70 % clone a 3-10 s window of another clip of the same speaker (the `ref` layout).
- 10 % drop the tag.

**Equivalence checks:**
- The sequence layouts match the upstream packer (`tests/test_pack.py`).
- The training forward matches the upstream forward (`tests/test_forward.py`) but runs the LocDiT loss only where it
  counts: about 2,100 positions/s on an A10, 2.3x upstream.
- Tagged text tokenizes exactly as an inference engine reads the exported tokenizer.

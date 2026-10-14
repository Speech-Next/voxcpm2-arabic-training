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
| `arabic_voxcpm.eval` | held-out evaluation sets, generation (library, dq_engine, Habibi), scoring, paired reports, blind A/B pages; `eval.run`: all in one command | #2 |
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

The sets are held out from training: speakers, clip ids and texts are listed in their `heldout.json`.

| set | texts per group | items | voices |
|---|---|---:|---:|
| `v1` | 30 (17 dialect groups) | 510 | 26 held-out speakers |
| `v2` | v1's 30 + 10 unseen: 40, except iq and ma (30) | 660 | v1's |

v2 keeps v1's items, ids and voices. Its new texts come from clips no training run could use: other clips of the
held-out lahgtna speakers, other SADA test-split clips, and for `sa` SADA test clips of Saudi speakers. A text that
any trainable or synthetic clip also has is left out. `iq` and `ma` have no such source: their lahgtna speakers are
in training. `v2/heldout.json` is v1's plus the new clips, so future training should exclude it.

Cloning modes (`--mode`): `ref` (timbre of the reference clip), `prompt_ref` (also its transcript, from
`voices.json`), `none` (no cloning; no speaker similarity). A run has one mode, recorded in `run.json`, and reports
compare it with a baseline of the same mode and set.

Metrics, per utterance:
- Whisper large-v3 and Cohere CER / WER after Arabic normalization, and the per-clip best of the two;
- UTMOS and DNSMOS;
- WavLM-SV similarity to the cloned voice;
- dialect region (ADI-5);
- stops: duration ratio (generated / expected at the group's real speaking rate), late stop (> 1.5),
  runaway (> 2 or `max_patches` reached), early stop (< 0.5). An empty generation counts as a full ASR error;
- per set, the ASR's error on the real recordings: the floor any model is compared against.

Reports are per group, with paired bootstrap 95 % intervals against the baselines (2 seeds by default).

One command generates, scores and reports a configuration. It runs in the scoring environment; generation runs in
the engine's environment (`ENGINE_PY`, `ENGINE`):

```bash
python -m arabic_voxcpm.eval.run --name <run> --model <dir | hf id> [--dit none | <ckpt>] [--dit-cache light] \
    [--precision bf16 | int8] [--mode ref | prompt_ref | none] [--set v1 | v2] [--baseline base-adapters ...] \
    [--seeds 2000 2001] [--engine dq] [generate.py options, e.g. --head-tokens 8]
# -> /workspace/data/arabic/eval/runs/<run>/{wav/,run.json,scores.jsonl,report.md,report.json}
```

- Steps that are already done are skipped. A run directory that holds another configuration is refused.
- A baseline name resolves to `<name>-<set>-<mode>`, `<name>-<mode>`, `<name>-<set>` or `<name>`, whichever matches
  the run's set and mode. Name baseline runs that way, for example `base-adapters-v2-prompt_ref`.
- Adapters: a tagged model uses the set's `dialect_map.json`, the base model each item's adapter. A group with no
  adapter uses the base model. An adapter the model does not have is an error.
- `--engine dq` generates with the base inference (dq_engine, `generate_dq.py`, `DQ_PY` / `DQ_ENGINE_DIR`). Its
  2.0.4 build has no seeded noise, so each seed is a replicate.

The steps on their own:

```bash
python -m arabic_voxcpm.eval.build --out /workspace/data/arabic/eval/v1
python -m arabic_voxcpm.eval.build --extend v1 --out /workspace/data/arabic/eval/v2 [--dry-run]   # counts per group
PYTHONPATH=.:<inference repo> <inference python> -m arabic_voxcpm.eval.generate --set v2 --out runs/<name> \
    --model <model> --mode ref
python -m arabic_voxcpm.eval.score runs/<name> [--gt /workspace/data/arabic/eval/v2]
python -m arabic_voxcpm.eval.score --fill-only runs/<name>          # add newer metrics to old rows (the stops: no model)
python -m arabic_voxcpm.eval.report --base runs/<baseline> [--base runs/<baseline-none>] runs/<name> --md r.md --json r.json
```

Blind listening test for native speakers:

```bash
python -m arabic_voxcpm.eval.listen --a runs/<A> --b runs/<B> --out ab/ [--groups eg najdi] [--n-per-group 10] [--seed 0]
# share ab/page/ (index.html + audio/; no external resources). Keep ab/key.json and ab/items.csv to yourself.
python -m arabic_voxcpm.eval.listen --score answers-*.csv --key ab/key.json   # win rates per group, Wilson 95 %
```

The page shows the Arabic text right to left, with players A and B. For "more natural" and "more like the dialect"
the listener picks A, B or same. Answers are kept in the browser and downloaded as CSV.

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

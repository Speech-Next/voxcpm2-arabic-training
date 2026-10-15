"""SADA (Saudi Audio Dataset for Arabic, SDAIA + Saudi Broadcasting Authority) -> latent shards + a manifest.

    python -m arabic_voxcpm.data.sada --out /workspace/data/arabic/sada

Audio: MohamedRashad/SADA22 (parquet, 16 kHz segments); per-segment metadata (show, file, speaker, environment,
dialect, segment id): khaledalganem/sada2022, the same rows in the same order (checked on the transcript of every
row). Kept: single-speaker segments of 1.5-20 s in a clean environment (--env adds others). Speaker = file +
the corpus's per-file speaker label. The test split is kept as FLAC (eval_audio/) for the evaluation (#2).
License: CC BY-NC-SA 4.0.

    python -m arabic_voxcpm.data.sada --restore <uids or dataset manifest.jsonl> --splits validation train

Latents only, as lahgtna.py --restore: the listed clips of the existing manifests, nothing else rewritten; only the
audio files holding a listed clip without latents are downloaded (the others' row counts are read remotely).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import time

import pyarrow.parquet as pq

from arabic_voxcpm import text as artext
from arabic_voxcpm import vae as avae
from arabic_voxcpm.data import common
from arabic_voxcpm.data.lahgtna import decode

AUDIO_REPO, META_REPO = "MohamedRashad/SADA22", "khaledalganem/sada2022"
SPLITS = {"train": "train", "validation": "validation", "test": "test"}
DIALECTS = {"Najdi": "najdi", "Hijazi": "hejazi", "Khaliji": "khaliji", "ModernStandardArabic": "msa",
            "Egyptian": "eg", "Levantine": "levantine", "Yemeni": "ye", "Shamali": "shamali", "Janubi": "janubi",
            "Iraqi": "iq", "Maghrebi": "maghrebi", "Unknown": "sa_unknown"}


def audio_files(split: str) -> list[str]:
    from huggingface_hub import list_repo_files

    return sorted(f for f in list_repo_files(AUDIO_REPO, repo_type="dataset") if f.startswith(f"data/{split}-"))


def metadata(split: str, raw: str) -> list[dict]:
    path = common.download(META_REPO, f"default/{split}/0000.parquet", raw, revision="refs/convert/parquet")
    return pq.read_table(path).to_pylist()


def same_text(audio_text, meta_text) -> bool:
    """The two copies agree; a missing transcript is None in the metadata and "nan" in the audio parquet."""
    return audio_text == meta_text or (meta_text is None and audio_text in (None, "", "nan"))


def keep_row(m: dict, envs: tuple[str, ...]) -> bool:
    return (m["SpeakerDialect"] in DIALECTS and m["Environment"].startswith(envs)
            and 1.5 <= m["SegmentLength"] <= 20 and bool(artext.clean(m["GroundTruthText"] or "")))


def restore(args) -> None:
    from huggingface_hub import HfFileSystem

    want = common.read_uids(args.restore)
    envs = tuple(args.env)
    raw = os.path.join(args.out, "raw")
    fs, vae = HfFileSystem(), None
    for split in args.splits:
        files = audio_files(split)
        todo = [(k, f, rows) for k, f in enumerate(files)
                if (rows := common.to_restore(args.out, f"{split}-{k:05d}", want))]
        hours = sum(r["frames"] for *_, rows in todo for r in rows) * avae.HOP / avae.SAMPLE_RATE / 3600
        clips = sum(len(t[2]) for t in todo)
        print(f"[{split}] {len(todo)} of {len(files)} files to restore: {clips} clips, {hours:.2f} h", flush=True)
        if not todo:
            continue
        meta = metadata(split, raw)
        sizes = [pq.ParquetFile(fs.open(f"datasets/{AUDIO_REPO}/{f}")).metadata.num_rows for f in files]
        assert sum(sizes) == len(meta), f"{split}: {sum(sizes)} audio rows, {len(meta)} metadata rows"
        offsets = [sum(sizes[:k]) for k in range(len(files))]
        if vae is None:
            avae.exact()
            vae = avae.load_vae(args.model)
        with cf.ThreadPoolExecutor(args.fetchers) as fetch, cf.ThreadPoolExecutor(args.workers) as pool:
            get = lambda t: common.download(AUDIO_REPO, t[1], raw)  # noqa: E731
            for (k, f, rows), path in common.ahead(fetch, get, todo, args.fetchers):
                t0 = time.time()
                name = f"{split}-{k:05d}"
                pf = pq.ParquetFile(path)
                assert pf.metadata.num_rows == sizes[k], f"{f}: {pf.metadata.num_rows} rows, {sizes[k]} read remotely"
                table = pf.read(columns=["text", "audio"])
                texts = table.column("text").to_pylist()
                metas = meta[offsets[k] : offsets[k] + sizes[k]]
                kept = [i for i, m in enumerate(metas) if keep_row(m, envs)]
                pos = {metas[i]["SegmentID"]: i for i in kept}
                assert len(pos) == len(kept), f"{f}: a segment id twice"
                for r in rows:
                    assert r["uid"] in pos, f"{f}: {r['uid']} is not a kept row of the source file"
                    i = pos[r["uid"]]
                    assert same_text(texts[i], metas[i]["GroundTruthText"]), f"{f}: audio and metadata rows disagree"
                    assert artext.clean(metas[i]["GroundTruthText"]) == r["text"], f"{f} {r['uid']}: the text differs"
                audio = table.column("audio")
                waves = list(pool.map(decode, [audio[pos[r["uid"]]].as_py() for r in rows]))
                del table, audio
                stats = common.restore_shard(args.out, name, vae, rows, waves)
                os.remove(path)
                print(f"[{split} {k + 1}/{len(files)}] {stats['clips']} clips restored, {stats['hours']:.2f} h "
                      f"in {time.time() - t0:.0f} s", flush=True)


def main(args) -> None:
    common.out_dirs(args.out)
    if args.restore:
        return restore(args)
    raw = os.path.join(args.out, "raw")
    vae = avae.load_vae(args.model)
    envs = tuple(args.env)
    with cf.ThreadPoolExecutor(1) as fetch, cf.ThreadPoolExecutor(args.workers) as pool:
        for split in args.splits:
            meta = metadata(split, raw)
            files = audio_files(split)
            # global row offsets: each file's row count, in order
            offset = 0
            nxt = fetch.submit(common.download, AUDIO_REPO, files[0], raw)
            for k, f in enumerate(files):
                path = nxt.result()
                nxt = fetch.submit(common.download, AUDIO_REPO, files[k + 1], raw) if k + 1 < len(files) else None
                name = f"{split}-{k:05d}"
                pf = pq.ParquetFile(path)
                n = pf.metadata.num_rows
                if common.done(args.out, name):
                    offset += n
                    os.remove(path)
                    continue
                t0 = time.time()
                table = pf.read()
                texts = table.column("text").to_pylist()
                metas = meta[offset : offset + n]
                assert len(metas) == n and all(same_text(t, m["GroundTruthText"]) for t, m in zip(texts, metas)), \
                    f"{f}: audio and metadata rows disagree"
                idx = [i for i, m in enumerate(metas) if keep_row(m, envs)]
                audio = table.column("audio")
                waves = list(pool.map(decode, [audio[i].as_py() for i in idx]))
                del table, audio
                rows = []
                for i in idx:
                    m = metas[i]
                    rows.append({"uid": m["SegmentID"], "source": "sada", "split": split, "shard": name,
                                 "dialect": DIALECTS[m["SpeakerDialect"]], "speaker": f"{m['FileName']}|{m['Speaker']}",
                                 "gender": m["SpeakerGender"], "age": m["SpeakerAge"], "environment": m["Environment"],
                                 "show": m["ShowName"], "category": m["Category"], "text": artext.clean(m["GroundTruthText"]),
                                 "text_raw": m["GroundTruthText"], "duration_s": m["SegmentLength"]})
                keep = {r["uid"] for r in rows} if split == "test" else set()
                stats = common.write_shard(args.out, name, vae, rows, waves, keep)
                offset += n
                os.remove(path)
                print(f"[{split} {k + 1}/{len(files)}] {stats['clips']} of {n} clips kept, {stats['hours']:.2f} h "
                      f"in {time.time() - t0:.0f} s", flush=True)
            assert offset == len(meta), f"{split}: {offset} audio rows, {len(meta)} metadata rows"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="/workspace/models/VoxCPM2-Arabic-Dialects", help="for the AudioVAE")
    ap.add_argument("--out", default="/workspace/data/arabic/sada")
    ap.add_argument("--splits", nargs="+", default=["test", "validation", "train"])
    ap.add_argument("--env", nargs="+", default=["Clean"], help="environments kept (Clean, Music, Noisy, Car)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--restore", help="latents only, of these clips (uids, or a dataset manifest.jsonl)")
    ap.add_argument("--fetchers", type=int, default=2, help="--restore: files downloading ahead")
    main(ap.parse_args())

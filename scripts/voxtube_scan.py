"""VoxTube: which parquet files / row groups hold Arabic rows (metadata columns only)."""
import json, sys, collections
from concurrent.futures import ThreadPoolExecutor
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
fs = HfFileSystem()
files = sorted(fs.glob("datasets/voice-is-cool/voxtube/data/train-*.parquet"))
print(len(files), "files", flush=True)
def scan(path):
    for k in range(5):
        try:
            with fs.open(path, block_size=1 << 20) as f:
                pf = pq.ParquetFile(f)
                out = []
                for g in range(pf.num_row_groups):
                    t = pf.read_row_group(g, columns=["language", "spk_id", "video_id", "segment_id", "gender"]).to_pydict()
                    ar = [i for i, l in enumerate(t["language"]) if l == "arabic"]
                    if ar:
                        out.append({"file": path, "rg": g, "rows": len(t["language"]), "ar_idx": ar,
                                    "spk": [t["spk_id"][i] for i in ar], "video": [t["video_id"][i] for i in ar],
                                    "seg": [t["segment_id"][i] for i in ar], "gender": [t["gender"][i] for i in ar]})
                return out
        except Exception as e:
            err = e
    print("failed", path, err, flush=True)
    return []
res = []
with ThreadPoolExecutor(4) as ex:
    for i, r in enumerate(ex.map(scan, files)):
        res += r
        if i % 200 == 0:
            print(i, sum(len(x["ar_idx"]) for x in res), flush=True)
json.dump(res, open(sys.argv[1], "w"))
n = sum(len(x["ar_idx"]) for x in res)
print("arabic rows", n, "files", len({x["file"] for x in res}), "speakers", len({s for x in res for s in x["spk"]}))

"""Kaggle GPU job: encode all *test* records with the fine-tuned e5 model, then neural top-k S1 retrieval per country.

Inputs (private dataset aml2026-er-assets): e5_er/ (model), test_source{1,2,3}.parquet.
Outputs (/kaggle/working): emb_test.npy (float16, rows = concat(source1, source2, source3)), neural_pairs_test.parquet.
Mirrors src/ber/neural.py (encode + retrieve) so row ids match the AWS pipeline.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

IN = next(Path("/kaggle/input").rglob("test_source1.parquet")).parent
MODEL = next(Path("/kaggle/input").rglob("config_sentence_transformers.json")).parent
OUT = Path("/kaggle/working")
DIM, K = 384, 5
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def texts():
    import pyarrow.parquet as pq
    out = []
    for s in (1, 2, 3):
        t = pq.read_table(IN / f"test_source{s}.parquet", columns=["business_name", "business_address"]).to_pydict()
        out += [f"query: {a} | {c}" for a, c in zip(t["business_name"], t["business_address"])]
    return out


def worker(gpu: int, lo: int, hi: int):
    import torch
    from sentence_transformers import SentenceTransformer
    t = texts()[lo:hi]
    m = SentenceTransformer(str(MODEL), device="cuda")
    m.max_seq_length = 64
    m.half()
    e = np.load(OUT / "emb_test.npy", mmap_mode="r+")[lo:hi]
    step = 200_000
    for i in range(0, len(t), step):
        with torch.inference_mode():
            e[i:i + step] = m.encode(t[i:i + step], batch_size=512, normalize_embeddings=True,
                                     convert_to_numpy=True, show_progress_bar=False).astype(np.float16)
        log(f"gpu{gpu}: {i + step}/{len(t)}")
    e.flush()


def retrieve(n_total: int):
    import polars as pl
    import torch
    rec = pl.concat([pl.read_parquet(IN / f"test_source{s}.parquet", columns=["country"])
                     .with_columns(pl.lit(s).cast(pl.Int8).alias("src")) for s in (1, 2, 3)]).with_row_index("rid")
    assert rec.height == n_total
    emb = np.load(OUT / "emb_test.npy", mmap_mode="r")
    outs = []
    for country in rec["country"].unique().sort().to_list():
        r1 = rec.filter((pl.col("country") == country) & (pl.col("src") == 1))["rid"].to_numpy()
        r2 = rec.filter((pl.col("country") == country) & (pl.col("src") != 1))["rid"].to_numpy()
        S = torch.from_numpy(np.ascontiguousarray(emb[r1])).cuda()
        qs, ss, cs, rk = [], [], [], []
        for i in range(0, len(r2), 2048):
            Q = torch.from_numpy(np.ascontiguousarray(emb[r2[i:i + 2048]])).cuda()
            v, idx = torch.topk(Q @ S.T, K, dim=1)
            n = Q.shape[0]
            qs.append(np.repeat(r2[i:i + 2048], K)); ss.append(r1[idx.cpu().numpy().ravel()])
            cs.append(v.float().cpu().numpy().ravel()); rk.append(np.tile(np.arange(K, dtype=np.int8), n))
        outs.append(pl.DataFrame({"q": np.concatenate(qs).astype(np.uint32), "s1": np.concatenate(ss).astype(np.uint32),
                                  "emb_cos": np.concatenate(cs).astype(np.float32), "rank_e": np.concatenate(rk)}))
        log(f"retrieve {country}: S1={len(r1)} Q={len(r2)}")
        del S
        torch.cuda.empty_cache()
    pl.concat(outs).write_parquet(OUT / "neural_pairs_test.parquet")


if __name__ == "__main__":
    if len(sys.argv) == 1:  # match the local training environment (model saved with sentence-transformers 6.1.0)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers==6.1.0", "polars==1.44.2"],
                       check=True)
    if len(sys.argv) > 1:  # child process: python encode_test.py gpu lo hi
        worker(*map(int, sys.argv[1:4]))
        sys.exit(0)
    import torch
    n = len(texts())
    ngpu = max(1, torch.cuda.device_count())
    log(f"records {n}, gpus {ngpu}: {[torch.cuda.get_device_name(i) for i in range(ngpu)]}")
    bounds = np.linspace(0, n, ngpu + 1).astype(int)
    full = np.lib.format.open_memmap(OUT / "emb_test.npy", mode="w+", dtype=np.float16, shape=(n, DIM))
    del full
    procs = [subprocess.Popen([sys.executable, __file__, str(g), str(bounds[g]), str(bounds[g + 1])],
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}) for g in range(ngpu)]
    assert all(p.wait() == 0 for p in procs), "encoder process failed"
    log("encoded; retrieving")
    retrieve(n)
    log("done")

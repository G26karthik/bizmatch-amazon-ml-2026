"""Kaggle GPU job: neural cosine for the pruned *test* pairs, using emb_test.npy from the aml2026-encode-test output.

Inputs: kernel output of aml2026-encode-test (emb_test.npy), dataset aml2026-pruned-test (pruned_pairs_test.parquet).
Output: /kaggle/working/emb_cos_pruned_test.parquet (q, s1, emb_cos).
"""
from pathlib import Path

import numpy as np
import polars as pl
import torch

EMB = next(Path("/kaggle/input").rglob("emb_test.npy"))
PAIRS = next(Path("/kaggle/input").rglob("pruned_pairs_test.parquet"))
emb = np.load(EMB, mmap_mode="r")
p = pl.read_parquet(PAIRS, columns=["q", "s1"]).sort("q")
q, s = p["q"].to_numpy().astype(np.int64), p["s1"].to_numpy().astype(np.int64)
out = np.empty(len(q), np.float32)
step = 1_000_000
for i in range(0, len(q), step):
    a = torch.from_numpy(np.ascontiguousarray(emb[q[i:i + step]])).cuda().float()
    b = torch.from_numpy(np.ascontiguousarray(emb[s[i:i + step]])).cuda().float()
    out[i:i + step] = (a * b).sum(1).cpu().numpy()
    print(f"{min(i + step, len(q))}/{len(q)}", flush=True)
p.with_columns(pl.Series("emb_cos", out)).write_parquet("/kaggle/working/emb_cos_pruned_test.parquet")
print("done", p.height)

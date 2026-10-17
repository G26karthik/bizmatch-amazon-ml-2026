"""Score (text_a, text_b) pairs with a trained cross-encoder on a local GPU, resumable in chunks.

python scripts/ce_score_local.py --pairs fr_ce_texts.parquet --model kaggle/ce_out_base2/ce_model --out work/fr_ce/base2 --bs 512
Writes <out>/part_XXXX.parquet (q, s1, pce) per chunk and <out>/ce_scores_test.parquet when all chunks are done.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", type=Path, required=True)
ap.add_argument("--model", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--bs", type=int, default=512)
ap.add_argument("--chunk", type=int, default=100_000)
ap.add_argument("--maxlen", type=int, default=128)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
T0 = time.time()
d = pl.read_parquet(a.pairs)
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForSequenceClassification.from_pretrained(a.model, dtype=torch.float16).cuda().eval()
n_chunks = (d.height + a.chunk - 1) // a.chunk
for c in range(n_chunks):
    f = a.out / f"part_{c:04d}.parquet"
    if f.exists():
        continue
    x = d.slice(c * a.chunk, a.chunk)
    ta, tb = x["text_a"].to_list(), x["text_b"].to_list()
    order = np.argsort([len(p) + len(q) for p, q in zip(ta, tb)])
    p = np.empty(len(ta), np.float32)
    with torch.inference_mode():
        for i in range(0, len(order), a.bs):
            j = order[i:i + a.bs]
            enc = tok([ta[k] for k in j], [tb[k] for k in j], truncation="longest_first", max_length=a.maxlen,
                      padding=True, return_tensors="pt").to("cuda")
            p[j] = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy()
    x.select("q", "s1").with_columns(pl.Series("pce", p)).write_parquet(f)
    print(f"[{time.time() - T0:7.0f}s] chunk {c + 1}/{n_chunks}", flush=True)
parts = sorted(a.out.glob("part_*.parquet"))
if len(parts) == n_chunks:
    pl.concat([pl.read_parquet(x) for x in parts]).write_parquet(a.out / "ce_scores_test.parquet")
    print(f"done: {d.height} pairs -> {a.out / 'ce_scores_test.parquet'}", flush=True)

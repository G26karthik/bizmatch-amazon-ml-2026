"""Neural name+address encoder (runs on a local CUDA GPU).

Base model: intfloat/multilingual-e5-small (MIT license, 118M params). Fine-tuned with in-batch-negative contrastive
loss on training match pairs (S1 record <-> matching S2/S3 record), validation groups excluded (same hash rule as
the main pipeline). Uses raw text, so Devanagari/Bengali/Telugu names are seen natively by the multilingual model.

  python -m ber.neural train   --data local_data --out local_data/e5_er
  python -m ber.neural encode  --data local_data --model local_data/e5_er --split train|test
  python -m ber.neural retrieve --data local_data --split train|test --k 10      -> neural_pairs_{split}.parquet
  python -m ber.neural score   --data local_data --split train|test --pairs P    -> emb_cos for given (q, s1) rids

Row ids (rid) follow the worker's records.parquet order: concat(source1, source2, source3) rows as stored.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl

SEED, VAL_MOD = 42, 10
DIM = 384
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def records(data: Path, split: str) -> pl.DataFrame:
    dfs = [pl.read_parquet(data / f"{split}_source{s}.parquet").with_columns(pl.lit(s).cast(pl.Int8).alias("src"))
           for s in (1, 2, 3)]
    return pl.concat(dfs).with_row_index("rid")


def text(df: pl.DataFrame) -> list[str]:
    return df.select((pl.lit("query: ") + pl.col("business_name") + " | " + pl.col("business_address")).alias("t"))["t"].to_list()


def train(data: Path, out: Path, n_pairs: int, epochs: int, bs: int) -> None:
    import torch
    from sentence_transformers import SentenceTransformer, losses, InputExample
    from torch.utils.data import DataLoader
    rec = records(data, "train")
    pairs = pl.read_parquet(data / "train_pairs.parquet")
    s1_val = rec.filter((pl.col("src") == 1) & (pl.col("entity_id").hash(SEED) % VAL_MOD == 0))["entity_id"]
    log(f"val S1 count {s1_val.len()} (worker reference: 220829)")
    assert s1_val.len() == 220829, "polars hash differs from worker -> validation split mismatch"
    tr = pairs.filter(~pl.col("s1").is_in(s1_val.implode())).sample(n_pairs, seed=SEED)
    t = rec.select("entity_id", (pl.lit("query: ") + pl.col("business_name") + " | " + pl.col("business_address")).alias("t"))
    tr = tr.join(t.rename({"entity_id": "s1", "t": "a"}), on="s1").join(t.rename({"entity_id": "m", "t": "b"}), on="m")
    ex = [InputExample(texts=[a, b]) for a, b in zip(tr["a"].to_list(), tr["b"].to_list())]
    log(f"training pairs {len(ex)}")
    model = SentenceTransformer("intfloat/multilingual-e5-small", device="cuda")
    model.max_seq_length = 64
    dl = DataLoader(ex, shuffle=True, batch_size=bs, drop_last=True)
    loss = losses.MultipleNegativesRankingLoss(model, scale=30.0)
    model.fit(train_objectives=[(dl, loss)], epochs=epochs, warmup_steps=300, use_amp=True, show_progress_bar=True,
              optimizer_params={"lr": 5e-5})
    model.save(str(out))
    log(f"saved {out}; GPU peak mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


def encode(data: Path, model_dir: Path, split: str, bs: int) -> None:
    """Streams 200k-row batches from each source file (low RAM: laptop has ~15 GB). Resumable: skips finished rows."""
    import pyarrow.parquet as pq
    import torch
    from sentence_transformers import SentenceTransformer
    files = [data / f"{split}_source{s}.parquet" for s in (1, 2, 3)]
    n = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    path, done_path = data / f"emb_{split}.npy", data / f"emb_{split}.done"
    done = int(done_path.read_text()) if done_path.exists() and path.exists() else 0
    emb = np.lib.format.open_memmap(path, mode="r+" if done else "w+", dtype=np.float16, shape=(n, DIM))
    model = SentenceTransformer(str(model_dir), device="cuda")
    model.max_seq_length = 64
    model.half()
    i = 0
    for f in files:
        for b in pq.ParquetFile(f).iter_batches(batch_size=200_000, columns=["business_name", "business_address"]):
            if i + b.num_rows <= done:
                i += b.num_rows
                continue
            d = b.to_pydict()
            t = [f"query: {a} | {c}" for a, c in zip(d["business_name"], d["business_address"])]
            with torch.inference_mode():
                e = model.encode(t, batch_size=bs, convert_to_numpy=True, normalize_embeddings=True,
                                 show_progress_bar=False)
            emb[i:i + len(e)] = e.astype(np.float16)
            i += len(e)
            emb.flush()
            done_path.write_text(str(i))
            log(f"encode {split}: {i}/{n}")
    del emb


def retrieve(data: Path, split: str, k: int, hide: int) -> None:
    """hide: percent of training S1 entities excluded from the pool (must equal the worker run's hide_pct)."""
    import torch
    rec = records(data, split).select("rid", "src", "country", "entity_id")
    if hide:
        rec = rec.filter(~((pl.col("src") == 1) & ((pl.col("entity_id").hash(SEED + 77) % 100) < hide)))
    emb = np.load(data / f"emb_{split}.npy", mmap_mode="r")
    outs = []
    for country in rec["country"].unique().sort().to_list():
        r1 = rec.filter((pl.col("country") == country) & (pl.col("src") == 1))["rid"].to_numpy()
        r2 = rec.filter((pl.col("country") == country) & (pl.col("src") != 1))["rid"].to_numpy()
        S = torch.from_numpy(np.ascontiguousarray(emb[r1])).cuda()
        step = 1024
        qs, ss, cs, rk = [], [], [], []
        for i in range(0, len(r2), step):
            Q = torch.from_numpy(np.ascontiguousarray(emb[r2[i:i + step]])).cuda()
            v, idx = torch.topk(Q @ S.T, k, dim=1)
            n = Q.shape[0]
            qs.append(np.repeat(r2[i:i + step], k)); ss.append(r1[idx.cpu().numpy().ravel()])
            cs.append(v.float().cpu().numpy().ravel()); rk.append(np.tile(np.arange(k, dtype=np.int8), n))
        outs.append(pl.DataFrame({"q": np.concatenate(qs).astype(np.uint32), "s1": np.concatenate(ss).astype(np.uint32),
                                  "emb_cos": np.concatenate(cs).astype(np.float32), "rank_e": np.concatenate(rk)}))
        log(f"retrieve {split}/{country}: S1={len(r1)} Q={len(r2)}")
        del S
        torch.cuda.empty_cache()
    pl.concat(outs).write_parquet(data / f"neural_pairs_{split}.parquet")


def score(data: Path, split: str, pairs_path: Path, out_path: Path) -> None:
    """emb_cos for arbitrary (q, s1) rid pairs, e.g. the worker's retrieval union."""
    import torch
    emb = np.load(data / f"emb_{split}.npy", mmap_mode="r")
    p = pl.read_parquet(pairs_path, columns=["q", "s1"]).sort("q")
    q, s = p["q"].to_numpy(), p["s1"].to_numpy()
    out = np.empty(len(q), np.float32)
    step = 500_000  # small chunks: laptop RAM is tight
    for i in range(0, len(q), step):
        a = torch.from_numpy(np.ascontiguousarray(emb[q[i:i + step]])).cuda().float()
        b = torch.from_numpy(np.ascontiguousarray(emb[s[i:i + step]])).cuda().float()
        out[i:i + step] = (a * b).sum(1).cpu().numpy()
        log(f"score {split}: {min(i + step, len(q))}/{len(q)}")
    p.with_columns(pl.Series("emb_cos", out)).write_parquet(out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train", "encode", "retrieve", "score"])
    ap.add_argument("--data", type=Path, default=Path("local_data"))
    ap.add_argument("--model", type=Path, default=Path("local_data/e5_er"))
    ap.add_argument("--out", type=Path, default=Path("local_data/e5_er"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--pairs", type=Path)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--hide", type=int, default=19)
    ap.add_argument("--n_pairs", type=int, default=1_000_000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=256)
    a = ap.parse_args()
    if a.stage == "train":
        train(a.data, a.out, a.n_pairs, a.epochs, a.bs)
    elif a.stage == "encode":
        encode(a.data, a.model, a.split, 512)
    elif a.stage == "retrieve":
        retrieve(a.data, a.split, a.k, a.hide if a.split == "train" else 0)
    else:
        score(a.data, a.split, a.pairs, a.out)

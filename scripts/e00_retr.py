"""E00 part 2: rebuild the TF-IDF retrieval (ber.pipeline.retrieve, exact same docs / IDF / DF cap / weights) for the
records of /work/agent_e00/need_retr.parquet (half-B true pairs missing from the pruned set) plus 3000 control pairs,
at depth KD, and record the rank of the true S1 under each retriever; e5 membership from neural_pairs_train.parquet.

retrieved (as in production) = rank_c < 10 | rank_a < 15 | rank_n < (40 if the record's address is empty else 8) | in e5.
-> /work/agent_e00/retr_ranks.parquet (q, s1, rank_c, rank_a, rank_n, in_e5, retrieved, min_rank, cos_a, cos_n, kth scores)
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn

sys.path.insert(0, "src")
from ber.pipeline import (K_ADDR, K_COMB, K_NAME, K_NAME_EMPTY, W_ADDR, W_NAME, _docs, _tfidf, hide_pct,  # noqa: E402
                          hidden_s1)

T0 = time.time()
OUT = Path("/work/agent_e00")
work = Path("/work/run28")
KD = 300
need = pl.concat([pl.read_parquet(OUT / "need_retr.parquet").with_columns(pl.lit(False).alias("ctl")),
                  pl.read_parquet(OUT / "ctl_retr.parquet").with_columns(pl.lit(True).alias("ctl"))]).with_columns(
    pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
rec = pl.read_parquet(work / "train" / "records.parquet",
                      columns=["entity_id", "src", "country", "name_n", "addr_n", "name_sk"]).with_row_index("rid")
pct = hide_pct(work)
outs, nb_out = [], []
for country in ["US", "India"]:
    r1 = rec.filter((pl.col("country") == country) & (pl.col("src") == 1))
    if pct:
        r1 = r1.filter(~hidden_s1(pl.col("entity_id"), pct))
    r2 = rec.filter((pl.col("country") == country) & (pl.col("src") != 1))
    nd = need.filter(pl.col("q").is_in(r2["rid"].cast(pl.UInt32).implode()))
    if nd.height == 0:
        continue
    print(f"[{time.time() - T0:5.0f}s] {country}: S1 {r1.height} Q {r2.height} need {nd.height}", flush=True)
    a1, n1 = _docs(r1)
    a2, n2 = _docs(r2)
    A1, A2, A1c = _tfidf(a1, a2)
    N1, N2, N1c = _tfidf(n1, n2)
    del a1, a2, n1, n2
    print(f"[{time.time() - T0:5.0f}s] tfidf done", flush=True)
    pos2 = np.full(int(rec.height), -1, np.int64)
    pos2[r2["rid"].to_numpy()] = np.arange(r2.height)
    pos1 = np.full(int(rec.height), -1, np.int64)
    pos1[r1["rid"].to_numpy()] = np.arange(r1.height)
    uq = np.unique(nd["q"].to_numpy())
    rows = pos2[uq]
    Qa, Qn = A2[rows], N2[rows]
    wa, wn = math.sqrt(W_ADDR), math.sqrt(W_NAME)
    CT = sp.hstack([A1c * wa, N1c * wn]).T.tocsr()
    Qc = sp.hstack([Qa * wa, Qn * wn]).tocsr()
    AT, NT = A1c.T.tocsr(), N1c.T.tocsr()
    empty = np.diff(Qa.indptr) == 0
    qidx = {int(q): i for i, q in enumerate(uq)}
    res = {}
    for nm, Q, TT in (("rank_c", Qc, CT), ("rank_a", Qa, AT), ("rank_n", Qn, NT)):
        C = sp_matmul_topn(Q, TT, top_n=KD, threshold=0.01, sort=True, n_threads=2).tocsr()
        res[nm] = C
        print(f"[{time.time() - T0:5.0f}s] {nm} top{KD} done", flush=True)
    ranks = {nm: [] for nm in res}
    kth = {nm: [] for nm in res}
    for q, s in zip(nd["q"].to_list(), nd["s1"].to_list()):
        i = qidx[q]
        col = pos1[s]
        for nm, C in res.items():
            a, b = C.indptr[i], C.indptr[i + 1]
            idx = C.indices[a:b]
            hit = np.flatnonzero(idx == col)
            ranks[nm].append(int(hit[0]) if len(hit) else 999)
            k = {"rank_c": K_COMB, "rank_a": K_ADDR, "rank_n": K_NAME}[nm]
            kth[nm].append(float(C.data[a + k - 1]) if b - a >= k else 0.0)
    # top-50 neighbours (comb and name retrievers) of the missing records, as competitor sets for the recoverability test
    rid1 = r1["rid"].to_numpy()
    for nm in ("rank_c", "rank_n"):
        C = res[nm]
        cnts = np.minimum(np.diff(C.indptr), 50)
        qq = np.repeat(uq, cnts)
        cc = np.concatenate([C.indices[C.indptr[i]:C.indptr[i] + cnts[i]] for i in range(len(uq))]) if len(uq) else np.array([], int)
        nb_out.append(pl.DataFrame({"q": qq.astype(np.uint32), "c": rid1[cc].astype(np.uint32)}).with_columns(pl.lit(nm).alias("retr")))
    qi, si = pos2[nd["q"].to_numpy()], pos1[nd["s1"].to_numpy()]
    cos_a = np.asarray(A2[qi].multiply(A1[si]).sum(1)).ravel()
    cos_n = np.asarray(N2[qi].multiply(N1[si]).sum(1)).ravel()
    o = nd.with_columns(*[pl.Series(nm, v, dtype=pl.Int32) for nm, v in ranks.items()],
                        *[pl.Series(f"kth_{nm[5:]}", v, dtype=pl.Float32) for nm, v in kth.items()],
                        pl.Series("q_empty_addr", empty[[qidx[q] for q in nd["q"].to_list()]]),
                        pl.Series("cos_a", cos_a.astype(np.float32)), pl.Series("cos_n", cos_n.astype(np.float32)),
                        pl.lit(country).alias("country"))
    outs.append(o)
    del A1, A2, A1c, N1, N2, N1c, CT, Qc, AT, NT, res
o = pl.concat(outs)
neu = pl.read_parquet(OUT / "neural_pairs_train.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
print("neural pairs columns", neu.columns, neu.height, flush=True)
o = o.join(neu.select("q", "s1", pl.lit(True).alias("in_e5")), on=["q", "s1"], how="left").with_columns(pl.col("in_e5").fill_null(False))
o = o.with_columns(((pl.col("rank_c") < K_COMB) | (pl.col("rank_a") < K_ADDR)
                    | (pl.col("rank_n") < pl.when(pl.col("q_empty_addr")).then(K_NAME_EMPTY).otherwise(K_NAME)) | pl.col("in_e5")).alias("retrieved"),
                   pl.min_horizontal("rank_c", "rank_a", "rank_n").alias("min_rank"))
o.write_parquet(OUT / "retr_ranks.parquet")
pl.concat(nb_out).unique(["q", "c"]).write_parquet(OUT / "retr_neighbours.parquet")
c = o.filter(pl.col("ctl"))
print(f"controls (true pairs in pruned set): retrieved by reconstruction {c['retrieved'].mean():.4f} (should be ~1)")
n = o.filter(~pl.col("ctl"))
print(f"missing pairs: {n.height}, retrieved (=> pruned by Stage A at pA<0.001) {n['retrieved'].sum()}, not retrieved {(~n['retrieved']).sum()}")
for K in (10, 20, 50, 100, 200, 300):
    print(f"  not-retrieved pairs found within min rank < {K}: {n.filter(~pl.col('retrieved') & (pl.col('min_rank') < K)).height}")
print(n.filter(~pl.col("retrieved")).group_by("country", "q_empty_addr").agg(pl.len(), (pl.col("rank_n") < 300).mean().alias("name_top300"),
      (pl.col("rank_n") < 100).mean().alias("name_top100"), (pl.col("rank_c") < 100).mean().alias("comb_top100")))
print(f"[{time.time() - T0:5.0f}s] done")

"""E00 part 6: would a different NAME retriever find the never-retrieved half-B copies?

Variants for the name retriever (same docs/IDF as ber.pipeline otherwise), rank of the true S1 at depth 300:
  base      words + char3 + skeleton, S1 DF cap 3000 (production)
  nocap     same docs, no DF cap
  bigram    docs + word bigrams ("allied_guild"), DF cap 3000
Also: exact normalised-name blocking (pool S1s with identical name_n). Output /work/agent_e00/retr2.parquet + summary.
"""
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sparse_dot_topn import sp_matmul_topn

sys.path.insert(0, "src")
from ber.pipeline import _docs, _tfidf, hide_pct, hidden_s1  # noqa: E402

T0 = time.time()
OUT = Path("/work/agent_e00")
NB = 89043
work = Path("/work/run28")
KD = 300
e = pl.read_parquet(OUT / "err_records.parquet").filter(pl.col("cat") == "FN_not_retrieved")
need = e.select("q", "s1", "loss", "acc", "q_addr_empty", "recov")
rec = pl.read_parquet(work / "train" / "records.parquet", columns=["entity_id", "src", "country", "name_n", "addr_n", "name_sk"]).with_row_index("rid")
pct = hide_pct(work)


def bigram_docs(df):
    out = []
    for nm, sk in zip(df["name_n"].to_list(), df["name_sk"].to_list()):
        w = nm.split()
        s = "_" + nm.replace(" ", "_") + "_"
        out.append(" ".join(w + [f"{x}_{y}" for x, y in zip(w, w[1:])] + ["#" + s[i:i + 3] for i in range(len(s) - 2)] + ["~" + k for k in sk.split()]))
    return out


def rank_of(Q, TT, qrows_for_pairs, cols):
    C = sp_matmul_topn(Q, TT, top_n=KD, threshold=0.01, sort=True, n_threads=2).tocsr()
    r = []
    for i, c in zip(qrows_for_pairs, cols):
        idx = C.indices[C.indptr[i]:C.indptr[i + 1]]
        h = np.flatnonzero(idx == c)
        r.append(int(h[0]) if len(h) else 999)
    return r


outs = []
for country in ["US", "India"]:
    r1 = rec.filter((pl.col("country") == country) & (pl.col("src") == 1))
    if pct:
        r1 = r1.filter(~hidden_s1(pl.col("entity_id"), pct))
    r2 = rec.filter((pl.col("country") == country) & (pl.col("src") != 1))
    nd = need.filter(pl.col("q").is_in(r2["rid"].cast(pl.UInt32).implode()))
    if nd.height == 0:
        continue
    pos2 = np.full(int(rec.height), -1, np.int64)
    pos2[r2["rid"].to_numpy()] = np.arange(r2.height)
    pos1 = np.full(int(rec.height), -1, np.int64)
    pos1[r1["rid"].to_numpy()] = np.arange(r1.height)
    uq = np.unique(nd["q"].to_numpy())
    qi = {int(q): i for i, q in enumerate(uq)}
    rows = [qi[q] for q in nd["q"].to_list()]
    cols = pos1[nd["s1"].to_numpy()]
    res = {}
    _, n1 = _docs(r1)
    _, n2 = _docs(r2)
    N1, N2, N1c = _tfidf(n1, n2)
    del n1, n2
    Q = N2[pos2[uq]]
    res["rank_base"] = rank_of(Q, N1c.T.tocsr(), rows, cols)
    res["rank_nocap"] = rank_of(Q, N1.T.tocsr(), rows, cols)
    print(f"[{time.time() - T0:5.0f}s] {country} base/nocap done", flush=True)
    del N1, N2, N1c
    b1, b2 = bigram_docs(r1), bigram_docs(r2)
    B1, B2, B1c = _tfidf(b1, b2)
    del b1, b2
    res["rank_bigram"] = rank_of(B2[pos2[uq]], B1c.T.tocsr(), rows, cols)
    del B1, B2, B1c
    print(f"[{time.time() - T0:5.0f}s] {country} bigram done", flush=True)
    outs.append(nd.with_columns(*[pl.Series(k, v, dtype=pl.Int32) for k, v in res.items()], pl.lit(country).alias("country")))
o = pl.concat(outs)
o.write_parquet(OUT / "retr2.parquet")
cfg = pl.Config(tbl_rows=40, tbl_width_chars=200)
for emp in (True, False):
    d = o.filter(pl.col("q_addr_empty") == emp)
    print(f"\n=== never-retrieved half-B copies, address empty={emp}: n={d.height}, loss {d['loss'].sum() / NB:.6f} ===")
    for K in (8, 40, 100, 300):
        row = [f"K={K:3d}"]
        for v in ("rank_base", "rank_nocap", "rank_bigram"):
            h = d.filter(pl.col(v) < K)
            row.append(f"{v[5:]}: n={h.height:4d} loss={h['loss'].sum() / NB:.6f} rec_est={(h['loss'] * h['acc'].fill_null(0)).sum() / NB:.6f}")
        print(" | ".join(row))
d = o.filter(pl.col("q_addr_empty"))
for v in ("rank_nocap", "rank_bigram"):
    x = d.filter(pl.col(v) < 40).group_by("recov").agg(pl.len(), (pl.col("loss").sum() / NB).alias("F05"),
                                                     ((pl.col("loss") * pl.col("acc").fill_null(0)).sum() / NB).alias("rec_est"))
    print(f"\n{v} < 40 (empty address) by recoverability:")
    with cfg:
        print(x.sort("F05", descending=True))
print(f"[{time.time() - T0:5.0f}s] done")

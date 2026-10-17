"""E00 part 8: cost of an uncapped name retriever for empty-address records (count on test, timing on a train sample)."""
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sparse_dot_topn import sp_matmul_topn

sys.path.insert(0, "src")
from ber.pipeline import _docs, _tfidf, hide_pct, hidden_s1  # noqa: E402

te = pl.read_parquet("/work/run28/test/records.parquet", columns=["src", "country", "addr_empty"])
print(te.filter(pl.col("src") != 1).group_by("country").agg(pl.len(), pl.col("addr_empty").sum().alias("empty")).sort("country"))
work = Path("/work/run28")
rec = pl.read_parquet(work / "train" / "records.parquet", columns=["entity_id", "src", "country", "name_n", "addr_n", "name_sk", "addr_empty"]
                      ).with_row_index("rid").filter(pl.col("country") == "US")
r1 = rec.filter((pl.col("src") == 1) & ~hidden_s1(pl.col("entity_id"), hide_pct(work)))
r2 = rec.filter(pl.col("src") != 1)
_, n1 = _docs(r1)
_, n2 = _docs(r2)
N1, N2, N1c = _tfidf(n1, n2)
idx = np.flatnonzero(r2["addr_empty"].to_numpy())
rng = np.random.default_rng(0)
s = rng.choice(idx, 20000, replace=False)
Q = N2[s]
for nm, M in (("capped", N1c), ("uncapped", N1)):
    T = M.T.tocsr()
    t0 = time.time()
    C = sp_matmul_topn(Q, T, top_n=40, threshold=0.01, sort=True, n_threads=2)
    dt = time.time() - t0
    print(f"{nm}: 20k empty-address queries top-40 in {dt:.1f}s with 2 threads -> {dt / 20000 * 1e6:.0f} us/query; nnz {C.nnz}")

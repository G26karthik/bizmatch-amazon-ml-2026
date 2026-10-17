"""Diagnostic: how well each cross-encoder score separates true/false pairs on the test-like validation uncertain zone
(unc_val_text pairs, H records removed). AUC of production prob, of each CE logit, and of CE vs prob on subsets.
python scripts/fast/ce_auc.py ceb2zs tlce_b ...
"""
import sys

import numpy as np
import polars as pl

SEED, HIDE = 42, 19
ids = pl.concat([pl.read_parquet(f"local_data/train_source{s}.parquet", columns=["entity_id", "business_address"]) for s in (1, 2, 3)]).with_row_index("rid")
tp = pl.read_parquet("local_data/train_pairs.parquet")
hid = ids.head(pl.read_parquet("local_data/train_source1.parquet", columns=["entity_id"]).height).filter(
    (pl.col("entity_id").hash(SEED + 77) % 100) < HIDE).select(pl.col("entity_id").alias("s1"))
hm = set(tp.join(hid, on="s1")["m"].to_list())
e2p = dict(tp.select("m", "s1").iter_rows())
r2e = ids["entity_id"].to_list()
emp = (ids["business_address"].fill_null("").str.strip_chars() == "").to_list()
u = pl.read_parquet("local_data/llm/unc_val_text.parquet")
q, s = u["q"].to_list(), u["s1"].to_list()
y = np.array([e2p.get(r2e[a]) == r2e[b] for a, b in zip(q, s)], dtype=np.int8)
keep = np.array([r2e[a] not in hm for a in q])
empty = np.array([emp[a] for a in q])


def auc(sc, yy):
    o = np.argsort(sc, kind="stable"); r = np.empty(len(sc)); r[o] = np.arange(1, len(sc) + 1)
    n1 = yy.sum(); n0 = len(yy) - n1
    return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


p = u["prob"].to_numpy()
lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
print(f"test-like pairs {keep.sum()} (of {len(y)}), pos rate {y[keep].mean():.3f}, empty-address {empty[keep].mean():.3f}")
print(f"{'score':10s} {'AUC all':>8s} {'AUC empty':>9s} {'AUC addr':>8s}  {'AUC(lp+z)':>9s}")
for nm, sc in [("prod_prob", lp)] + [(n, u.select("q", "s1").join(pl.read_parquet(f"local_data/llm/{n}_val.parquet"), on=["q", "s1"], how="left")["logit"].to_numpy()) for n in sys.argv[1:]]:
    k = keep & ~np.isnan(sc)
    comb = lp + 0.3 * sc if nm != "prod_prob" else lp
    print(f"{nm:10s} {auc(sc[k], y[k]):8.4f} {auc(sc[k & empty], y[k & empty]):9.4f} {auc(sc[k & ~empty], y[k & ~empty]):8.4f}  {auc(comb[k], y[k]):9.4f}")

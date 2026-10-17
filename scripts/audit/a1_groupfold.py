"""Audit: record-fold vs S1-group-fold held-out band precision for the agent_tb namesake and big one-op tie-breakers."""
import sys, zlib
import lightgbm as lgb, numpy as np, polars as pl
WHICH = sys.argv[1]; sys.argv = sys.argv[:1]
sys.path.insert(0, "scripts/agent_tb")
from tb_model import add_group_feats, top_picks
import tb_model_ns as NS

which = WHICH
if which == "ns":
    tr = pl.read_parquet("local_data/work/tb_ns_train.parquet").with_columns(big=pl.lit(True))
    tr = add_group_feats(NS.fold_col(tr)).with_columns(lncf=pl.col("ncand_full").log1p())
    tr = NS.extra(tr); META = NS.META
else:
    big = pl.read_parquet("local_data/work/tb_big_train.parquet").with_columns(big=pl.lit(True))
    small = pl.read_parquet("local_data/work/tb_pairs_train.parquet").with_columns(ncand_full=pl.col("ncand"), big=pl.lit(False))
    tr = pl.concat([big, small.select(big.columns)], how="vertical_relaxed")
    tr = add_group_feats(NS.fold_col(tr)).with_columns(lncf=pl.col("ncand_full").log1p())
    META = ["m", "s1", "country", "kind", "label", "has_truth", "fr27_s1", "fr28_s1", "fold", "big"]
feats = [c for c in tr.columns if c not in META and c not in ("gfold",)]
# group fold: connected by min candidate s1 of the record (namesake groups share candidate sets)
g = tr.group_by("m").agg(mins=pl.col("s1").min())
g = g.with_columns(gfold=pl.col("mins").map_elements(lambda x: zlib.crc32(x.encode()) % 2, return_dtype=pl.Int64))
tr = tr.join(g.select("m", "gfold"), on="m", how="left")
params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=14, seed=7)
X = tr.select(feats).to_numpy(); y = tr["label"].cast(pl.Int8).to_numpy()
res = {}
for fc in ("fold", "gfold"):
    fold = tr[fc].to_numpy(); oof = np.zeros(tr.height)
    for f in (0, 1):
        b = lgb.train(params, lgb.Dataset(X[fold == f], y[fold == f], feature_name=feats, categorical_feature=["op_id", "ctry", "src"]), num_boost_round=500)
        oof[fold != f] = b.predict(X[fold != f])
    t = top_picks(tr.filter(pl.col("big")).with_columns(p=pl.Series(oof[tr["big"].to_numpy()])), "p")
    res[fc] = t
    print("==", which, fc, "top1", round(t["label"].mean(), 4))
    for c in ("US", "India"):
        tc = t.filter(pl.col("country") == c)
        for lo, hi in ((0.75, 0.8), (0.8, 0.85), (0.85, 0.9), (0.9, 0.95), (0.95, 1.01)):
            s = tc.filter((pl.col("p") >= lo) & (pl.col("p") < hi))
            print(f"  {c} [{lo},{hi}) n {s.height} prec {s['label'].mean():.4f}")
res["gfold"].select("m", "s1", "p", "label", "country").write_parquet(f"local_data/work/audit/gfold_{which}.parquet")

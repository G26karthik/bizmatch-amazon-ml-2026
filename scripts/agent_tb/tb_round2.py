"""Round 2: retrain tie-breakers with raw-name copy-proxy features (small one-op, big one-op, namesake cap 120).
Same crc32 halves. Per-country threshold = lowest band edge where every band at/above has precision >= 0.87 in both
halves (bands with n >= 30). Writes local_data/work/tb_r2_<name>_top.parquet (test top picks with p and thr).

uv run ... python scripts/agent_tb/tb_round2.py small|big|ns120
"""
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

NAME = sys.argv.pop(1)
sys.path.insert(0, "scripts/agent_tb")
from tb_model import add_group_feats, top_picks  # noqa: E402
from tb_model_ns import extra, fold_col  # noqa: E402

pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_width_chars(250)
META = ["m", "s1", "country", "kind", "label", "has_truth", "fr27_s1", "fr28_s1", "fr35_s1", "fr44_s1", "fold", "big"]
EDGES = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]
W = "local_data/work/"


def load(name, split):
    if name == "ns120":
        return pl.read_parquet(f"{W}tb_ns120_{split}.parquet")
    f = {"small": "tb_pairs_", "big": "tb_big_"}[name] + split
    d = pl.read_parquet(f"{W}{f}.parquet").join(pl.read_parquet(f"{W}{f}_raw.parquet"), on=["m", "s1"], how="left")
    if name == "small":
        d = d.with_columns(ncand_full=pl.col("ncand"))
    return d


def main(name):
    tr = load(name, "train").with_columns(big=pl.lit(True))
    if name == "big":   # as tb_model_big: fit on big + small groups
        sm = load("small", "train").with_columns(big=pl.lit(False))
        tr = pl.concat([tr, sm.select(tr.columns)], how="vertical_relaxed")
    tr = extra(add_group_feats(fold_col(tr)).with_columns(lncf=pl.col("ncand_full").log1p()))
    feats = [c for c in tr.columns if c not in META]
    X = tr.select(feats).to_numpy()
    y = tr["label"].cast(pl.Int8).to_numpy()
    fold = tr["fold"].to_numpy()
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=14, seed=7)
    oof = np.zeros(tr.height)
    models = []
    for f in (0, 1):
        b = lgb.train(params, lgb.Dataset(X[fold == f], y[fold == f], feature_name=feats, categorical_feature=["op_id", "ctry", "src"]),
                      num_boost_round=500 if name != "small" else 700)
        models.append(b)
        oof[fold != f] = b.predict(X[fold != f])
    tr = tr.with_columns(p=pl.Series(oof))
    t = top_picks(tr.filter(pl.col("big")), "p")
    print(f"[{name}] train records {t.height}, top-1 acc {t['label'].mean():.4f}")
    thr = {}
    for c in ("US", "India"):
        tc = t.filter(pl.col("country") == c)
        print(f"  {c}: records {tc.height}")
        for a, b_ in zip(EDGES[:-1], EDGES[1:]):
            s = [tc.filter((pl.col("p") >= a) & (pl.col("p") < b_) & (pl.col("fold") == f)) for f in (0, 1)]
            sp = tc.filter((pl.col("p") >= a) & (pl.col("p") < b_))
            cu = tc.filter(pl.col("p") >= a)
            print(f"    band [{a},{min(b_, 1.0)}): n {sp.height} prec {sp['label'].mean() if sp.height else float('nan'):.4f} "
                  f"(h0 {s[0]['label'].mean() if s[0].height else float('nan'):.4f}, h1 {s[1]['label'].mean() if s[1].height else float('nan'):.4f})"
                  f" | p>={a}: cov {cu.height / tc.height:.4f} prec {cu['label'].mean() if cu.height else float('nan'):.4f}")
        for lo in EDGES[:-1]:
            ok = True
            for a, b_ in zip(EDGES[:-1], EDGES[1:]):
                if a < lo:
                    continue
                for f in (0, 1):
                    s = tc.filter((pl.col("p") >= a) & (pl.col("p") < b_) & (pl.col("fold") == f))
                    if s.height >= 30 and s["label"].mean() < 0.87:
                        ok = False
            if ok:
                thr[c] = lo
                break
    print(f"  chosen per-country thresholds: {thr}")
    te = load(name, "test")
    te = extra(add_group_feats(te).with_columns(lncf=pl.col("ncand_full").log1p()))
    Xt = te.select(feats).to_numpy()
    te = te.with_columns(p=pl.Series((models[0].predict(Xt) + models[1].predict(Xt)) / 2))
    tt = top_picks(te, "p").with_columns(thr=pl.col("country").replace_strict(thr, default=9.0, return_dtype=pl.Float64),
                                         model=pl.lit(name))
    tt.select("m", "s1", "country", "kind", "p", "thr", "model").write_parquet(f"{W}tb_r2_{name}_top.parquet")
    print(f"  test top picks written: {tt.height}")


if __name__ == "__main__":
    main(NAME)

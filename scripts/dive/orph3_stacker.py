"""Replica of the v27 stacker (ber.blend gbm: 4 CEs + raw zone features, 2-fold cross-fit on half-A S1 groups) with the
hidden-parent-orphan (H) rows of the training set dropped or down-weighted.  Evaluates half B:
  standard  : as ber.blend (rule chosen on half A with orphan FPs x3; report plain k=1 and hard k=3)
  test-like : H records removed from the candidate pool (they do not exist on test), rule chosen on half A under the same
              metric; raw-orphan FPs counted k=1 and k=2 (test has ~2x the raw-orphan density of train)
Writes only /work/dive/orph/stk_<variant>_*.parquet.
python orph3_stacker.py --variant base|noH|w0.3
"""
import argparse, json, os
from pathlib import Path
import numpy as np, polars as pl
import lightgbm as lgb
from ber.pipeline import SEED, labels, _best_per_query
from ber.stage2 import _chosen

ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="base")
a = ap.parse_args()
OUT = Path("/work/dive/orph"); W = Path("/work/run24")
CES = [Path(p) for p in ["/work/ce_wide", "/work/ce_wide_base", "/work/ce_wide_base2", "/work/ce_large"]]
RAW = Path("/work/rawf")
NT = 2


def _logit(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


lab = labels(Path("/work/data"), Path("/work/base")).with_columns(
    (pl.col("parent").fill_null(pl.col("rid")).hash(SEED + 7) % 2 == 0).alias("half_a"))
ha = lab.select(pl.col("rid").cast(pl.UInt32).alias("s1"), pl.col("half_a").alias("s1_half_a"))
rawp = pl.read_parquet(OUT / "rawp.parquet")
Hset = lab.filter(pl.col("parent").is_null() & (pl.col("src") != 1)).select(pl.col("rid").cast(pl.UInt32).alias("q")) \
          .join(rawp.select(pl.col("q").cast(pl.UInt32)), on="q", how="semi")
print("H records (all train):", Hset.height, flush=True)


def attach(df, split):
    for i, c in enumerate(CES):
        s = pl.read_parquet(c / f"ce_scores_{split}.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32)).rename({"pce": f"pce{i}"})
        df = df.join(s, on=["q", "s1"], how="left")
    r = pl.read_parquet(RAW / f"raw_{split}.parquet")
    df = df.join(r.rename({c: f"raw_{c}" for c in r.columns if c not in ("q", "s1")}), on=["q", "s1"], how="left")
    return df.with_columns(pl.col("pce0").alias("pce"))


def feats(df):
    d = df.with_columns((pl.col("p3") - pl.col("p3").max().over("q")).alias("_m3"))
    cols = [_logit(d["p3"].to_numpy())]
    for i in range(len(CES)):
        c = f"pce{i}"
        d = d.with_columns(pl.col(c).fill_null(0.0).alias("_c"))
        d = d.with_columns(pl.col("_c").rank("ordinal", descending=True).over("q").alias("_r"), pl.col("_c").max().over("q").alias("_top"))
        second = d.filter(pl.col("_r") == 2).select("q", pl.col("_c").alias("_sec"))
        d = d.drop([x for x in ("_sec",) if x in d.columns]).join(second, on="q", how="left")
        d = d.with_columns(pl.when(pl.col("_r") == 1).then(pl.col("_c") - pl.col("_sec").fill_null(0.0))
                           .otherwise(pl.col("_c") - pl.col("_top")).alias("_marg"))
        cols += [_logit(d[c].fill_null(0.5).to_numpy()), d["_marg"].to_numpy(), (d["_r"] == 1).to_numpy().astype(float)]
    cols.append(d["_m3"].to_numpy())
    return np.column_stack(cols)


def feats_gbm(df):
    d = df.with_columns(pl.len().over("q").alias("_nq"), pl.len().over("s1").alias("_ns"),
                        pl.col("p3").rank("ordinal", descending=True).over("q").alias("_r3"), pl.col("p3").sum().over("s1").alias("_s3"))
    extra = [d["_nq"].to_numpy(), d["_ns"].to_numpy(), d["_r3"].to_numpy(), d["_s3"].to_numpy()]
    for i in range(len(CES)):
        c = pl.col(f"pce{i}").fill_null(0.0)
        e = d.select(c.sum().over("s1").alias("a"), (c > 0.5).sum().over("s1").alias("b"), (c > 0.5).sum().over("q").alias("c"),
                     pl.col(f"pce{i}").is_null().cast(pl.Float32).alias("n"))
        extra += [e[x].to_numpy() for x in ("a", "b", "c", "n")]
    extra += [df[c].cast(pl.Float32).to_numpy() for c in df.columns if c.startswith("raw_")]
    return np.column_stack([feats(df)] + extra).astype(np.float32)


v = attach(pl.read_parquet(W / "scores_val.parquet"), "val").join(ha, on="s1", how="left")
isH = v["q"].is_in(Hset["q"].implode()).to_numpy()
s1va = lab.filter((pl.col("src") == 1) & pl.col("is_val") & pl.col("half_a"))["rid"].cast(pl.UInt32)
fitmask = (v["pce"].is_not_null() & v["s1"].is_in(s1va.implode())).to_numpy()
y = v["y"].to_numpy()
X = feats_gbm(v)
wt = np.ones(len(y))
orph_q = lab.filter(pl.col("parent").is_null() & (pl.col("src") != 1)).select(pl.col("rid").cast(pl.UInt32).alias("q"))
isR = v["q"].is_in(orph_q["q"].implode()).to_numpy() & ~isH
if a.variant == "noH":
    fitmask = fitmask & ~isH
elif a.variant == "noHR2":  # test-like: no absent-business copies, twice the raw-orphan (decoy) density
    fitmask = fitmask & ~isH
    wt[isR] = 2.0
elif a.variant.startswith("w"):
    wt[isH] = float(a.variant[1:])
print(f"variant {a.variant}: fit rows {fitmask.sum()}, H rows among zone half-A {int((isH & (v['pce'].is_not_null() & v['s1'].is_in(s1va.implode())).to_numpy()).sum())}", flush=True)
prm = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=200, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=SEED, num_threads=NT)
gfold = (v["s1"].hash(SEED + 11) % 2).to_numpy()
oof, models = np.full(v.height, np.nan), []
for k in range(2):
    tr, va = fitmask & (gfold != k), fitmask & (gfold == k)
    m = lgb.train(prm, lgb.Dataset(X[tr], y[tr], weight=wt[tr]), 3000, valid_sets=[lgb.Dataset(X[va], y[va], weight=wt[va])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    oof[va] = m.predict(X[va], num_iteration=m.best_iteration)
    models.append(m)
    print(f"fold {k}: iters {m.best_iteration}", flush=True)
pv = np.mean([m.predict(X, num_iteration=m.best_iteration) for m in models], axis=0)
pv[fitmask] = oof[fitmask]
# H rows of half A that were excluded from fitting: give them the fold-average (they are not used for fitting anyway)
prob = np.where(v["pce"].is_null().to_numpy(), v["p3"].to_numpy(), pv).astype(np.float32)
v = v.with_columns(pl.Series("prob", prob))
v.select("q", "s1", "prob").write_parquet(OUT / f"stk_{a.variant}_val_pairs.parquet")
best = _best_per_query(v)
best.write_parquet(OUT / f"stk_{a.variant}_val_best.parquet")

# ---- evaluation
par = lab.filter(pl.col("src") != 1).select(pl.col("rid").cast(pl.UInt32).alias("q"), pl.col("parent").cast(pl.UInt32),
                                             pl.col("parent").is_null().alias("orph"))
par = par.with_columns(pl.col("q").is_in(Hset["q"].implode()).alias("H"))
ntrue = lab.filter(pl.col("parent").is_not_null()).group_by(pl.col("parent").cast(pl.UInt32).alias("s1")).agg(pl.len().alias("nt"))
s1a = lab.filter((pl.col("src") == 1) & pl.col("is_val") & pl.col("half_a")).select(pl.col("rid").cast(pl.UInt32).alias("s1")).join(ntrue, on="s1", how="left").fill_null(0)
s1b = lab.filter((pl.col("src") == 1) & pl.col("is_val") & ~pl.col("half_a")).select(pl.col("rid").cast(pl.UInt32).alias("s1")).join(ntrue, on="s1", how="left").fill_null(0)


def macro(ch, s1df, k):
    c = ch.join(par, on="q", how="left").with_columns((pl.col("parent") == pl.col("s1")).fill_null(False).alias("tp"))
    c = c.with_columns(pl.when(pl.col("tp")).then(0).when(pl.col("orph")).then(k).otherwise(1).alias("fp"))
    g = c.group_by("s1").agg(pl.col("tp").sum().cast(pl.Float64), pl.col("fp").sum().cast(pl.Float64))
    e = s1df.join(g, on="s1", how="left").fill_null(0)
    f = e.select(pl.when(pl.col("nt") == 0).then((pl.col("fp") == 0).cast(pl.Float64))
                 .otherwise(1.25 * pl.col("tp") / (1.25 * pl.col("tp") + pl.col("fp") + 0.25 * (pl.col("nt") - pl.col("tp")))).alias("f"))
    return float(f["f"].mean())


grid = [{"mode": "thr", "thr": round(float(t), 2)} for t in np.arange(0.5, 0.96, 0.05)] + \
       [{"mode": "efopt", "miss": 0.0, "gamma": g} for g in [0.8, 1.0, 1.25, 1.5, 2.0, 3.0]]


def evaluate(best, tag):
    out = {}
    for world, b in (("standard", best), ("testlike", best.filter(~pl.col("q").is_in(Hset["q"].implode())))):
        chs = {json.dumps(c): _chosen(b, c) for c in grid}
        for kk in ((1, 3) if world == "standard" else (1, 2)):
            ca = max(grid, key=lambda c: macro(chs[json.dumps(c)], s1a, kk))
            ch = chs[json.dumps(ca)]
            out[f"{world}_k{kk}"] = dict(cfg=ca, valB_k1=round(macro(ch, s1b, 1), 5), valB_kk=round(macro(ch, s1b, kk), 5))
        # fixed production rule (gamma 1.5)
        ch = chs[json.dumps({"mode": "efopt", "miss": 0.0, "gamma": 1.5})]
        out[f"{world}_g1.5"] = dict(valB_k1=round(macro(ch, s1b, 1), 5), valB_k2=round(macro(ch, s1b, 2), 5), valB_k3=round(macro(ch, s1b, 3), 5))
    print(tag, json.dumps(out), flush=True)
    return out


res = {"variant": a.variant, "iters": [m.best_iteration for m in models], "stacked": evaluate(best, f"stacker_{a.variant}")}
if a.variant == "base":
    prod = pl.read_parquet(W / "val_best_run24_both.parquet")
    res["production_v27"] = evaluate(prod, "production_v27")
json.dump(res, open(OUT / f"stk_{a.variant}_report.json", "w"), indent=1)

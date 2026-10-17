"""Evaluate the LoRA LLM judge. Uncertain best-S1 pairs (0.02 < p < 0.98) get a combined probability from a logistic model
on [logit(p), llm_logit, product]; fitted on validation half A, scored on half B with the production decision rule
(efopt, gamma 1.5), paired bootstrap vs the current blend (plain and hard metric). With --write the model is refitted
on all validation pairs and applied to test -> /work/run28/{val,test}_best_run28_7j.parquet + val_report_run28_7j.json.

python scripts/eval_judge.py [--write] [--control]
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, "src")
from ber.metric import entity_f05  # noqa: E402
from ber.pipeline import SEED, labels  # noqa: E402
from ber.stage2 import _chosen  # noqa: E402

work, tag = Path("/work/run28"), "run28_7q"
cfg = {"mode": "efopt", "miss": 0.0, "gamma": 1.5}
lab = labels(Path("/work/data"), work).with_columns((pl.col("parent").fill_null(pl.col("rid")).hash(SEED + 7) % 2 == 0).alias("half_a"))
half = dict(lab.filter(pl.col("src") == 1).select("rid", "half_a").iter_rows())
S = {h: lab.filter((pl.col("src") == 1) & pl.col("is_val") & (pl.col("half_a") == (h == "A")))["rid"].to_list() for h in "AB"}
parent = dict(lab.filter(pl.col("parent").is_not_null()).select("rid", "parent").iter_rows())
truth = {}
for q, p in parent.items():
    truth.setdefault(p, set()).add(q)
orph = set(lab.filter(pl.col("parent").is_null() & (pl.col("src") != 1))["rid"].to_list())
vb = pl.read_parquet(work / f"val_best_{tag}.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
lv = pl.read_parquet("/work/dive/llm_val.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
v = vb.join(lv, on=["q", "s1"], how="left")
v = v.with_columns(pl.Series("y", [parent.get(q) == s for q, s in zip(v["q"].to_list(), v["s1"].to_list())]),
                   pl.Series("hA", [half.get(s, False) for s in v["s1"].to_list()]))


CONTROL = "--control" in sys.argv  # recalibration only: same logistic model without the LLM columns


def feats(df):
    p = np.clip(df["prob"].to_numpy(), 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    ll = df["logit"].to_numpy()
    return np.c_[lp] if CONTROL else np.c_[lp, ll, lp * ll]


unc = v.filter(pl.col("logit").is_not_null())
print("uncertain val pairs with LLM score:", unc.height, "half A", unc.filter(pl.col("hA")).height, flush=True)
yy = unc["y"].to_numpy().astype(int)
p0 = np.clip(unc["prob"].to_numpy(), 1e-6, 1 - 1e-6)
pl_ = 1 / (1 + np.exp(-unc["logit"].to_numpy()))
ll = lambda y, p: -np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))  # noqa: E731
print(f"logloss on uncertain val pairs: blend {ll(yy, p0):.4f}  llm {ll(yy, pl_):.4f}", flush=True)


def combine(fit_df, apply_df):
    m = LogisticRegression(C=1.0, max_iter=1000).fit(feats(fit_df), fit_df["y"].to_numpy().astype(int))
    return m, m.predict_proba(feats(apply_df))[:, 1]


mA, pB = combine(unc.filter(pl.col("hA")), unc.filter(~pl.col("hA")))
uB = unc.filter(~pl.col("hA"))
print(f"half B uncertain logloss: blend {ll(uB['y'].to_numpy().astype(int), uB['prob'].to_numpy()):.4f} -> combined {ll(uB['y'].to_numpy().astype(int), pB):.4f}; coef {mA.coef_.round(3)} {mA.intercept_.round(3)}")
newB = uB.select("q", "s1").with_columns(pl.Series("p_new", pB))
vb2 = vb.join(newB, on=["q", "s1"], how="left").with_columns(pl.coalesce("p_new", "prob").alias("prob")).drop("p_new")


def per_entity(best):
    ch = _chosen(best, cfg)
    plain, hard = {}, {}
    for q, s in ch.select("q", "s1").iter_rows():
        plain.setdefault(s, set()).add(q)
        hard.setdefault(s, set()).add(q)
        if q in orph:
            hard[s] |= {("d", q, 1), ("d", q, 2)}
    return {m: np.array([entity_f05(truth.get(s, ()), pm.get(s, ())) for s in S["B"]]) for m, pm in (("plain", plain), ("hard", hard))}


a, b = per_entity(vb), per_entity(vb2)
rng = np.random.default_rng(0)
for m in ("plain", "hard"):
    d = b[m] - a[m]; n = len(d)
    bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(2000)])
    print(f"half B {m}: {a[m].mean():.6f} -> {b[m].mean():.6f} diff {d.mean():+.6f} CI [{np.quantile(bs, .025):+.6f},{np.quantile(bs, .975):+.6f}] P(<=0) {np.mean(bs <= 0):.3f}")
if "--write" in sys.argv:
    mall, pv = combine(unc, unc)
    vball = vb.join(unc.select("q", "s1").with_columns(pl.Series("p_new", pv)), on=["q", "s1"], how="left") \
              .with_columns(pl.coalesce("p_new", "prob").alias("prob")).drop("p_new")
    vball.write_parquet(work / "val_best_run28_7j.parquet")
    tb = pl.read_parquet(work / f"test_best_{tag}.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
    lt = pl.read_parquet("/work/dive/llm_test.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
    t = tb.join(lt, on=["q", "s1"], how="inner")
    pt = mall.predict_proba(feats(t))[:, 1]
    tb2 = tb.join(t.select("q", "s1").with_columns(pl.Series("p_new", pt)), on=["q", "s1"], how="left") \
            .with_columns(pl.coalesce("p_new", "prob").alias("prob")).drop("p_new")
    tb2.write_parquet(work / "test_best_run28_7j.parquet")
    shutil.copy(work / f"val_report_{tag}.json", work / "val_report_run28_7j.json")
    print("wrote run28_7j (val/test best + report); test pairs rescored:", t.height)

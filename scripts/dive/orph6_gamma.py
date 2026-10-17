"""Decision strictness in the test-like world (hidden-parent orphans removed, raw-orphan FPs x k): gamma sweep on the
production v27 val_best, per-record gamma for empty-address vs other records, paired bootstrap on half B, and the
number of test pairs that change."""
import json
from pathlib import Path
import numpy as np, polars as pl
from ber.pipeline import SEED, labels
from ber.stage2 import _chosen
OUT = Path("/work/dive/orph"); W = Path("/work/run24"); TAG = "run24_both"
lab = labels(Path("/work/data"), Path("/work/base")).with_columns(
    (pl.col("parent").fill_null(pl.col("rid")).hash(SEED + 7) % 2 == 0).alias("half_a"))
vf = pl.read_parquet(OUT / "val_feats.parquet", columns=["q", "cls", "q_addr_empty", "q_country"])
Hq = vf.filter(pl.col("cls") == "H").select("q")
par = lab.filter(pl.col("src") != 1).select(pl.col("rid").cast(pl.UInt32).alias("q"), pl.col("parent").cast(pl.UInt32),
                                             pl.col("parent").is_null().alias("orph"))
ntrue = lab.filter(pl.col("parent").is_not_null()).group_by(pl.col("parent").cast(pl.UInt32).alias("s1")).agg(pl.len().alias("nt"))
S1 = {h: lab.filter((pl.col("src") == 1) & pl.col("is_val") & (pl.col("half_a") == (h == "A"))).select(pl.col("rid").cast(pl.UInt32).alias("s1"))
      .join(ntrue, on="s1", how="left").fill_null(0).sort("s1") for h in "AB"}


def per_s1(ch, h, k):
    c = ch.join(par, on="q", how="left").with_columns((pl.col("parent") == pl.col("s1")).fill_null(False).alias("tp"))
    c = c.with_columns(pl.when(pl.col("tp")).then(0).when(pl.col("orph")).then(k).otherwise(1).alias("fp"))
    g = c.group_by("s1").agg(pl.col("tp").sum().cast(pl.Float64), pl.col("fp").sum().cast(pl.Float64))
    e = S1[h].join(g, on="s1", how="left").fill_null(0).sort("s1")
    return e.select(pl.when(pl.col("nt") == 0).then((pl.col("fp") == 0).cast(pl.Float64))
                    .otherwise(1.25 * pl.col("tp") / (1.25 * pl.col("tp") + pl.col("fp") + 0.25 * (pl.col("nt") - pl.col("tp")))))\
            .to_series().to_numpy()


vb = pl.read_parquet(W / f"val_best_{TAG}.parquet").join(vf.select("q", "q_addr_empty"), on="q", how="left")
vtl = vb.join(Hq, on="q", how="anti")


def chosen2(b, ge, ga):
    x = b.with_columns(pl.when(pl.col("q_addr_empty")).then(pl.col("prob") ** ge).otherwise(pl.col("prob") ** ga).alias("prob"))
    return _chosen(x.select("q", "s1", "prob"), {"mode": "efopt", "miss": 0.0, "gamma": 1.0})


rows = []
for g in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5]:
    cs, ct = chosen2(vb, g, g), chosen2(vtl, g, g)
    rows.append(dict(ge=g, ga=g, std_A_k3=per_s1(cs, "A", 3).mean(), std_B_k1=per_s1(cs, "B", 1).mean(), std_B_k3=per_s1(cs, "B", 3).mean(),
                     tl_A_k2=per_s1(ct, "A", 2).mean(), tl_B_k1=per_s1(ct, "B", 1).mean(), tl_B_k2=per_s1(ct, "B", 2).mean()))
    print({k: (round(x, 5) if isinstance(x, float) else x) for k, x in rows[-1].items()}, flush=True)
for ge in [0.5, 0.7, 0.8, 1.0]:
    for ga in [0.8, 1.0, 1.5]:
        if ge == ga:
            continue
        ct = chosen2(vtl, ge, ga)
        rows.append(dict(ge=ge, ga=ga, tl_A_k2=per_s1(ct, "A", 2).mean(), tl_B_k1=per_s1(ct, "B", 1).mean(), tl_B_k2=per_s1(ct, "B", 2).mean()))
        print({k: (round(x, 5) if isinstance(x, float) else x) for k, x in rows[-1].items()}, flush=True)
r = pl.DataFrame(rows)
r.write_parquet(OUT / "gamma_sweep.parquet")
best = r.filter(pl.col("tl_A_k2").is_not_null()).sort("tl_A_k2", descending=True).row(0, named=True)
print("best on test-like half A (k2):", best, flush=True)
# paired bootstrap on test-like half B: best vs production gamma 1.5, and gamma 0.8 vs 1.5
rng = np.random.default_rng(0)
base = per_s1(chosen2(vtl, 1.5, 1.5), "B", 2)
for ge, ga in ((best["ge"], best["ga"]), (0.8, 0.8), (1.0, 1.0)):
    d = per_s1(chosen2(vtl, ge, ga), "B", 2) - base
    n = len(d); bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(1000)])
    print(f"test-like k2 half B ({ge},{ga}) vs 1.5: diff {d.mean():+.5f} CI [{np.quantile(bs, .025):+.5f},{np.quantile(bs, .975):+.5f}] P<=0 {np.mean(bs <= 0):.3f} better {int((d > 0).sum())} worse {int((d < 0).sum())}", flush=True)
# test pairs that change
cfg = json.load(open(W / f"val_report_{TAG}.json"))["best"]
tf = pl.read_parquet(OUT / "test_feats.parquet", columns=["q", "q_addr_empty", "q_country"])
tb = pl.read_parquet(W / f"test_best_{TAG}.parquet").join(tf, on="q", how="left")
prod = _chosen(tb.select("q", "s1", "prob"), cfg)
for ge, ga in ((best["ge"], best["ga"]), (0.8, 0.8)):
    new = chosen2(tb, ge, ga)
    add = new.join(prod, on=["q", "s1"], how="anti").join(tf, on="q")
    rem = prod.join(new, on=["q", "s1"], how="anti").join(tf, on="q")
    print(f"test ({ge},{ga}) vs production: pairs {prod.height} -> {new.height}; added {add.height}, removed {rem.height}", flush=True)
    print("  added by country x empty:", add.group_by("q_country", "q_addr_empty").len().sort("q_country", "q_addr_empty").rows())
    print("  removed by country x empty:", rem.group_by("q_country", "q_addr_empty").len().sort("q_country", "q_addr_empty").rows())

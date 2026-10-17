"""Model-implied (expected) macro F0.5 per country, to locate where the leaderboard loses points without labels.

For every S1 entity, the records whose best candidate is that S1 are true with their blended probability (independent
Bernoulli draws); the chosen set is fixed by the decision rule. Averaging F0.5 over draws gives the score the model
expects. On validation (US/India) the expectation is compared with the real score to check calibration; on test it is
reported per country, including France which has no labels.

python scripts/expected_f05.py --work /work/run20 --tag run20_large
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "src")
from ber.pipeline import SEED, labels  # noqa: E402
from ber.stage2 import _chosen  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, default=Path("/work/data"))
ap.add_argument("--work", type=Path, required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--draws", type=int, default=20)
a = ap.parse_args()
cfg = json.loads((a.work / f"val_report_{a.tag}.json").read_text())["best"]


def expected(best: pl.DataFrame, s1_ids: pl.DataFrame, draws: int) -> pl.DataFrame:
    """best: q, s1, prob (one row per record); s1_ids: s1, country. Returns per-S1 expected F0.5 plus country."""
    ch = _chosen(best, cfg).with_columns(pl.lit(True).alias("sel"))
    b = best.join(ch, on=["q", "s1"], how="left").with_columns(pl.col("sel").fill_null(False))
    rng = np.random.default_rng(SEED)
    p, sel = b["prob"].to_numpy().astype(np.float64), b["sel"].to_numpy()
    s1 = b["s1"].to_numpy()
    out = []
    for _ in range(draws):
        t = rng.random(len(p)) < p
        d = pl.DataFrame({"s1": s1, "tp": t & sel, "fp": ~t & sel, "fn": t & ~sel})
        g = d.group_by("s1").agg(pl.col("tp").sum(), pl.col("fp").sum(), pl.col("fn").sum())
        out.append(g)
    g = pl.concat(out).with_columns(
        pl.when(pl.col("tp") + pl.col("fn") == 0).then((pl.col("fp") == 0).cast(pl.Float64))
          .otherwise(1.25 * pl.col("tp") / (1.25 * pl.col("tp") + pl.col("fp") + 0.25 * pl.col("fn"))).alias("f"))
    g = g.group_by("s1").agg((pl.col("f").sum() / draws).alias("ef"), pl.len().alias("n"))
    # S1s that are in no draw (no record points to them) score 1 (empty truth, empty prediction)
    return s1_ids.join(g, on="s1", how="left").with_columns(
        (pl.col("ef") + (draws - pl.col("n").fill_null(0)) / draws).fill_null(1.0).alias("ef")).drop("n")


# validation: calibration check on half B (US / India)
lab = labels(a.data, a.work).with_columns(
    (pl.col("parent").fill_null(pl.col("rid")).hash(SEED + 7) % 2 == 0).alias("half_a"))
rec_tr = pl.read_parquet(a.work / "train" / "records.parquet", columns=["country"]).with_row_index("rid")
vb = lab.filter((pl.col("src") == 1) & pl.col("is_val") & ~pl.col("half_a")).select(pl.col("rid").cast(pl.UInt32).alias("s1")) \
        .join(rec_tr.select(pl.col("rid").cast(pl.UInt32).alias("s1"), "country"), on="s1")
bv = pl.read_parquet(a.work / f"val_best_{a.tag}.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
ev = expected(bv.filter(pl.col("s1").is_in(vb["s1"].implode())), vb, a.draws)
truth: dict = {}
for q, par in lab.filter(pl.col("parent").is_not_null()).select("rid", "parent").iter_rows():
    truth.setdefault(par, set()).add(q)
pm: dict = {}
for q, s in _chosen(bv, cfg).iter_rows():
    pm.setdefault(s, set()).add(q)
from ber.metric import entity_f05  # noqa: E402
ev = ev.with_columns(pl.Series("real", [entity_f05(truth.get(s, ()), pm.get(s, ())) for s in ev["s1"].to_list()]))
print("validation half B (expected vs real macro F0.5):")
print(ev.group_by("country").agg(pl.len(), pl.col("ef").mean().alias("expected"), pl.col("real").mean().alias("real")).sort("country"))

# test: per country
rec_te = pl.read_parquet(a.work / "test" / "records.parquet", columns=["src", "country"]).with_row_index("rid")
ts = rec_te.filter(pl.col("src") == 1).select(pl.col("rid").cast(pl.UInt32).alias("s1"), "country")
bt = pl.read_parquet(a.work / f"test_best_{a.tag}.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
et = expected(bt, ts, a.draws)
print("test (expected macro F0.5 by country):")
print(et.group_by("country").agg(pl.len(), pl.col("ef").mean().alias("expected")).sort("country"))
print("test overall expected:", round(float(et["ef"].mean()), 5))
q = bt.join(rec_te.select(pl.col("rid").cast(pl.UInt32).alias("q"), "country"), on="q")
print(q.group_by("country").agg(pl.len().alias("records"), ((pl.col("prob") > 0.1) & (pl.col("prob") < 0.9)).mean().alias("uncertain_0.1_0.9"),
                                (pl.col("prob") >= 0.9).mean().alias("confident")).sort("country"))

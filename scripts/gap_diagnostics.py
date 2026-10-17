"""Where does the validation -> leaderboard gap come from? Run on worker 1 (run4).

1) v4 validation F0.5 per country (US vs India) on val-B.
2) v4 test prediction statistics per country (France vs US vs India): share of queries assigned, confidence profile,
   matches per S1, empty S1 share. France behaving differently from US/India on test is the tell-tale sign.
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "src")
from ber.metric import entity_f05  # noqa: E402
from ber.pipeline import SEED, decide_pairs, labels  # noqa: E402

work = Path("/work/run4")
cfg = json.loads((work / "val_report_s2.json").read_text())["best"]
lab = labels(Path("/work/data"), work).with_columns(
    (pl.col("parent").fill_null(pl.col("rid")).hash(SEED + 7) % 2 == 0).alias("half_a"))
rec = pl.read_parquet(work / "train" / "records.parquet", columns=["country"]).with_row_index("rid")
best = pl.read_parquet(work / "val_best_s2.parquet")
chosen = decide_pairs(best, cfg.get("miss", 0.0)) if cfg["mode"] == "efopt" else best.filter(pl.col("prob") >= cfg["thr"]).select("q", "s1")
pred: dict = {}
for q, s in chosen.iter_rows():
    pred.setdefault(s, set()).add(q)
truth: dict = {}
for q, par in lab.filter(pl.col("parent").is_not_null()).select("rid", "parent").iter_rows():
    truth.setdefault(par, set()).add(q)
vb = lab.filter((pl.col("src") == 1) & pl.col("is_val") & ~pl.col("half_a")).select("rid").join(rec, on="rid")
for c in ("US", "India"):
    ids = vb.filter(pl.col("country") == c)["rid"].to_list()
    f = np.array([entity_f05(truth.get(s, ()), pred.get(s, ())) for s in ids])
    sing = np.array([len(truth.get(s, ())) == 0 for s in ids])
    print(f"valB {c}: n={len(ids)} F0.5={f.mean():.5f}  singletons={sing.mean():.3f} F(singleton)={f[sing].mean():.4f} "
          f"F(non-singleton)={f[~sing].mean():.4f}")

trec = pl.read_parquet(work / "test" / "records.parquet", columns=["country", "src"]).with_row_index("rid")
tb = pl.read_parquet(work / "test_best_s2.parquet").join(trec.select(pl.col("rid").alias("q"), "country"), on="q")
tch = decide_pairs(tb.select("q", "s1", "prob"), cfg.get("miss", 0.0)).join(trec.select(pl.col("rid").alias("q"), "country"), on="q")
nq = trec.filter(pl.col("src") != 1).group_by("country").len().rename({"len": "queries"})
ns1 = trec.filter(pl.col("src") == 1).group_by("country").len().rename({"len": "s1"})
st = (tb.group_by("country").agg(pl.len().alias("q_with_cand"), pl.col("prob").median().alias("med_p"),
                                 ((pl.col("prob") > 0.2) & (pl.col("prob") < 0.8)).mean().alias("uncertain"))
        .join(tch.group_by("country").len().rename({"len": "assigned"}), on="country")
        .join(tch.group_by("country").agg(pl.col("s1").n_unique().alias("s1_with_match")), on="country")
        .join(nq, on="country").join(ns1, on="country")
        .with_columns((pl.col("assigned") / pl.col("queries")).alias("assigned_frac"),
                      (pl.col("assigned") / pl.col("s1")).alias("matches_per_s1"),
                      (1 - pl.col("s1_with_match") / pl.col("s1")).alias("empty_s1_frac")))
with pl.Config(tbl_cols=20, tbl_width_chars=220):
    print(st.select("country", "queries", "s1", "assigned_frac", "matches_per_s1", "empty_s1_frac", "med_p", "uncertain"))
# same statistics for validation (hidden-S1 density) to compare with test
vrec = rec.join(lab.select("rid", "src", "is_val"), on="rid")
vq = vrec.filter((pl.col("src") != 1) & pl.col("is_val")).group_by("country").len().rename({"len": "queries"})
vs = vrec.filter((pl.col("src") == 1) & pl.col("is_val")).group_by("country").len().rename({"len": "s1"})
vch = chosen.join(rec.select(pl.col("rid").alias("q"), "country"), on="q").group_by("country").len().rename({"len": "assigned"})
print("validation (test-like density):",
      vq.join(vs, on="country").join(vch, on="country").with_columns(
          (pl.col("assigned") / pl.col("s1")).alias("matches_per_s1")).to_dicts())

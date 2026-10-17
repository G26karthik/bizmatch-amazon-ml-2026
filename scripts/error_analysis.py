"""Break down validation errors by type, country and singleton status. Run on worker after `train`."""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "src")
from ber.metric import entity_f05  # noqa: E402
from ber.pipeline import labels  # noqa: E402

data, work = Path("/work/data"), Path("/work/run")
thr = 0.65
lab = labels(data, work)
rec = pl.read_parquet(work / "train" / "records.parquet",
                      columns=["business_name", "business_address", "country", "name_nonascii", "addr_empty"]).with_row_index("rid")
best = pl.read_parquet(work / "val_best.parquet")  # q, s1, prob (argmax per query, s1 in val)
pairs = pl.read_parquet(work / "train" / "pairs.parquet", columns=["q", "s1"])
val_s1 = lab.filter((pl.col("src") == 1) & pl.col("is_val")).select(pl.col("rid").alias("s1"))
vq = lab.filter(pl.col("is_val") & (pl.col("src") != 1)).select(pl.col("rid").alias("q"), "parent")

# per true val pair outcome
retrieved = pairs.join(vq, on="q").filter(pl.col("s1") == pl.col("parent")).select("q").unique().with_columns(pl.lit(True).alias("ret"))
pred = best.filter(pl.col("prob") >= thr).select("q", pl.col("s1").alias("pred_s1"), "prob")
t = (vq.filter(pl.col("parent").is_not_null()).join(retrieved, on="q", how="left").join(pred, on="q", how="left")
       .with_columns(pl.when(pl.col("ret").is_null()).then(pl.lit("not_retrieved"))
                     .when(pl.col("pred_s1").is_null()).then(pl.lit("below_thr_or_argmax_elsewhere"))
                     .when(pl.col("pred_s1") == pl.col("parent")).then(pl.lit("correct"))
                     .otherwise(pl.lit("assigned_wrong_s1")).alias("outcome")))
t = t.join(rec.select(pl.col("rid").alias("q"), "country", "name_nonascii", "addr_empty"), on="q")
print("TRUE-PAIR OUTCOMES\n", t.group_by("outcome").len().sort("len", descending=True))
print(t.group_by("country", "outcome").len().sort("country", "len", descending=[False, True]))
print("by nonascii name\n", t.group_by("name_nonascii", "outcome").len().sort("name_nonascii", "len", descending=[False, True]))
print("by empty addr\n", t.group_by("addr_empty", "outcome").len().sort("addr_empty", "len", descending=[False, True]))

# false positives: predictions onto val S1 that are wrong (query has other/no parent)
fp = pred.join(vq.select("q", "parent").vstack(lab.filter(~pl.col("is_val") & (pl.col("src") != 1)).select(pl.col("rid").alias("q"), "parent")), on="q", how="left")
fp = fp.filter(pl.col("pred_s1").is_in(val_s1["s1"].implode()) & (pl.col("parent").is_null() | (pl.col("parent") != pl.col("pred_s1"))))
print("FALSE POSITIVES onto val S1:", fp.height, " of which query truly unmatched:", fp.filter(pl.col("parent").is_null()).height)

# per-entity F by country / singleton
truth = t.group_by("parent").agg(pl.col("q")).rename({"parent": "s1"})
pr = pred.group_by("pred_s1").agg(pl.col("q").alias("p")).rename({"pred_s1": "s1"})
e = val_s1.join(truth, on="s1", how="left").join(pr, on="s1", how="left").join(rec.select(pl.col("rid").alias("s1"), "country"), on="s1")
f = [entity_f05(a or [], b or []) for a, b in zip(e["q"].to_list(), e["p"].to_list())]
e = e.with_columns(pl.Series("f", f), (pl.col("q").list.len().fill_null(0) == 0).alias("singleton"))
print("macro", float(np.mean(f)))
print(e.group_by("country", "singleton").agg(pl.len(), pl.col("f").mean()).sort("country", "singleton"))

# samples of each error type
with pl.Config(tbl_rows=15, fmt_str_lengths=55, tbl_width_chars=230):
    for kind in ("not_retrieved", "below_thr_or_argmax_elsewhere", "assigned_wrong_s1"):
        s = t.filter(pl.col("outcome") == kind).sample(12, seed=1)
        s = (s.join(rec.select(pl.col("rid").alias("q"), pl.col("business_name").alias("q_name"), pl.col("business_address").alias("q_addr")), on="q")
              .join(rec.select(pl.col("rid").alias("parent"), pl.col("business_name").alias("true_name"), pl.col("business_address").alias("true_addr")), on="parent"))
        if kind == "assigned_wrong_s1":
            s = s.join(rec.select(pl.col("rid").alias("pred_s1"), pl.col("business_name").alias("pred_name"), pl.col("business_address").alias("pred_addr")), on="pred_s1")
            print(kind, "\n", s.select("q_name", "true_name", "pred_name", "q_addr", "true_addr", "pred_addr"))
        else:
            print(kind, "\n", s.select("q_name", "true_name", "q_addr", "true_addr", "prob"))
    fps = fp.filter(pl.col("parent").is_null()).sample(12, seed=2).join(
        rec.select(pl.col("rid").alias("q"), pl.col("business_name").alias("q_name"), pl.col("business_address").alias("q_addr")), on="q").join(
        rec.select(pl.col("rid").alias("pred_s1"), pl.col("business_name").alias("s1_name"), pl.col("business_address").alias("s1_addr")), on="pred_s1")
    print("FP unmatched query -> val S1\n", fps.select("q_name", "s1_name", "q_addr", "s1_addr", "prob"))

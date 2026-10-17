"""Model-free copy-or-decoy test. For S1s having a one-word-swap record of a category, count the S1's CERTAIN copies:
records with identical name key and identical address token set (99.99% true in train). An S1 with a copy of the
category is size-biased (more copies); an S1 next to a decoy is not (US truth: decoy S1s have fewer copies).
Reports P(no certain copy) and mean certain copies per category; US train truth calibrates.

python scripts/swap_ident_test.py
"""
import re
import sys
import unicodedata

import polars as pl

sys.argv = sys.argv[:1] + ["/work/out_fr7"]
src = open("/work/code/scripts/swap_verdict.py").read().split("def verdict")[0]
ns = {"__name__": "x"}
exec(src, ns)
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(200)
GEN_FR = ["fils", "cie", "associes", "services", "developpement", "groupe", "france", "frs", "freres", "compagnie"]
GEN_US = ["center", "services", "service", "partners", "centre"]


def ident_counts(rec):
    r = rec.with_columns(pl.col("nt").list.join(" ").alias("nk"),
                         pl.concat_str([pl.col("street"), pl.col("hn").cast(pl.String)], separator="#").alias("ak"))
    s1 = r.filter((pl.col("src") == 1) & (pl.col("street") != "")).select(pl.col("rid").alias("s1"), "country", "nk", "ak")
    q = r.filter((pl.col("src") != 1) & (pl.col("street") != "")).select("country", "nk", "ak")
    return s1.join(q, on=["country", "nk", "ak"]).group_by("s1").agg(pl.len().alias("n_ident"))


def report(d, idc, label):
    x = d.select("s1", "cat").unique().join(idc, on="s1", how="left").with_columns(pl.col("n_ident").fill_null(0))
    print(f"\n=== {label}")
    print(x.group_by("cat").agg(pl.len().alias("n_s1"), (pl.col("n_ident") == 0).mean().round(4).alias("p_no_certain_copy"),
                                pl.col("n_ident").mean().round(3).alias("mean_certain")).sort("cat"))


tr = ns["prep"]("/work/base/train/records.parquet").filter(pl.col("country") == "US")
d = ns["swaps"](tr)
gt = pl.read_parquet("/work/data/train_pairs.parquet").select(pl.col("s1").alias("eid_s1"), pl.col("m").alias("eid_q"), pl.lit(True).alias("true"))
d = d.join(gt, on=["eid_s1", "eid_q"], how="left").with_columns(pl.col("true").fill_null(False))
d = d.with_columns(pl.concat_str([pl.col("arel"), pl.when(pl.col("w_in").is_in(GEN_US)).then(pl.lit("GEN")).otherwise(pl.lit("OTHER")),
                                  pl.when(pl.col("true")).then(pl.lit("TRUE")).otherwise(pl.lit("FALSE"))], separator="|").alias("cat"))
report(d, ident_counts(tr), "US train truth")
te = ns["prep"]("/work/runfr/test/records.parquet")
f = pl.read_parquet("/work/dive/fr_swaps.parquet").with_columns(
    pl.concat_str([pl.col("arel"), pl.when(pl.col("w_in").is_in(GEN_FR)).then(pl.lit("GEN")).otherwise(pl.lit("TYPE"))], separator="|").alias("cat"))
idc = ident_counts(te.filter(pl.col("country") == "France"))
report(f, idc, "France test")
words = ["club", "ecole", "amicale", "comite", "sportive", "fils", "cie", "services", "associes", "groupe", "developpement", "france"]
report(f.filter(pl.col("w_in").is_in(words)).with_columns(pl.concat_str([pl.col("arel"), pl.col("w_in")], separator="|").alias("cat")), idc, "France per word")
fr_all = te.filter((pl.col("country") == "France") & (pl.col("src") == 1)).select(pl.col("rid").alias("s1"), pl.lit("ALL_FR_S1").alias("cat"))
report(fr_all, idc, "France all S1 (baseline)")

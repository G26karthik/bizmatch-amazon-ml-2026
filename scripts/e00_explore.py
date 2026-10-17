"""E00 part 3: profile the error records (/work/agent_e00/err_records.parquet) and print examples per bucket."""
import sys
from pathlib import Path

import polars as pl

OUT = Path("/work/agent_e00")
NB = 89043
e = pl.read_parquet(OUT / "err_records.parquet")
fn = e.filter(pl.col("cat").str.starts_with("FN"))
cols = ["cat", "country", "q_src", "T", "q_raw_name", "q_raw_addr", "s_raw_name", "s_raw_addr", "s_name_cnt", "q_name_cnt", "am_p",
        "p_sc", "o_sc", "recov", "loss"]
cfgp = pl.Config(tbl_rows=40, fmt_str_lengths=45, tbl_width_chars=400, tbl_cols=20)


def summ(df, by, title):
    g = df.group_by(by).agg(pl.len().alias("n"), (pl.col("loss").sum() / NB).alias("F05")).sort("F05", descending=True)
    print(f"\n--- {title} ---")
    with cfgp:
        print(g)


fn = fn.with_columns(
    pl.when(pl.col("q_name_cnt") == 0).then(pl.lit("0")).when(pl.col("q_name_cnt") == 1).then(pl.lit("1"))
      .when(pl.col("q_name_cnt") <= 3).then(pl.lit("2-3")).when(pl.col("q_name_cnt") <= 10).then(pl.lit("4-10"))
      .otherwise(pl.lit(">10")).alias("qn_cnt_b"),
    (pl.col("q_name") == pl.col("s_name")).alias("name_eq_parent"))
emp = fn.filter(pl.col("q_addr_empty"))
summ(emp, ["name_eq_parent", "qn_cnt_b"], "empty-address FN: record name == parent name x #pool S1s with the record's exact name")
summ(fn.filter(~pl.col("q_addr_empty")), ["cat", "recov"], "address FN: category x recoverability")
# empty address, name unique in pool and equal to parent -> should be easy
easy = emp.filter(pl.col("name_eq_parent") & (pl.col("q_name_cnt") == 1))
summ(easy, ["cat"], "empty-address FN with a UNIQUE exact parent name (pool count 1)")
with cfgp:
    print(easy.sort("loss", descending=True).select(cols).head(15))
# empty address, noisy name matching no S1 exactly
nz = emp.filter(pl.col("q_name_cnt") == 0)
summ(nz, ["cat", "recov"], "empty-address FN whose name matches no pool S1 exactly")
with cfgp:
    print(nz.filter(pl.col("recov") == "rec_parent_best_clear").sort("loss", descending=True).select(cols).head(12))
    print(nz.filter(pl.col("recov") == "irr_competitor_as_good").sort("loss", descending=True).select(cols + ["o_c"]).head(12))
# not chosen with clear best parent
nc = fn.filter((pl.col("cat") == "FN_not_chosen") & (pl.col("recov") == "rec_parent_best_clear"))
print("\nFN_not_chosen & parent clearly best: prob quantiles", [round(nc["am_p"].quantile(x), 3) for x in (0.1, 0.25, 0.5, 0.75, 0.9)])
with cfgp:
    print(nc.sort("loss", descending=True).select(cols).head(15))
# address records lost at stage A / not candidate with clear best parent
ad = fn.filter(~pl.col("q_addr_empty") & pl.col("cat").is_in(["FN_not_cand", "FN_pruned_A_0.001-0.01", "FN_not_retrieved", "FN_pruned_A<0.001"]))
with cfgp:
    print("\naddress records never scored, parent clearly best:")
    print(ad.filter(pl.col("recov") == "rec_parent_best_clear").sort("loss", descending=True).select(cols + ["pA"]).head(15))
# FP examples
fp = e.filter(pl.col("cat").str.starts_with("FP"))
with cfgp:
    print("\nFP examples:")
    print(fp.sort("loss", descending=True).select(["cat", "country", "q_raw_name", "q_raw_addr", "s_raw_name", "s_raw_addr", "am_p", "fp_sc", "loss"]).head(15))

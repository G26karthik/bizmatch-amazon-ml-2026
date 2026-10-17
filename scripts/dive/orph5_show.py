"""Print the saved group table (orph4) and sibling/frequency tables with all columns."""
from pathlib import Path
import polars as pl
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(400); pl.Config.set_fmt_str_lengths(30)
OUT = Path("/work/dive/orph")
tab = pl.read_parquet(OUT / "group_table.parquet")
keep = ["g", "v_n", "H_sh", "R_sh", "true_all", "true_noH", "v_mprob", "v_acc", "rej_true_noH", "rej_mprob_noH", "v_rej_noH",
        "acc_prec_noH", "t_n", "t_mprob", "t_acc", "t_rej", "t_rej_p.2"]
t = tab.select(keep)
print("groups with >= 300 val records and (true_noH - true_all) >= 0.02 or H_sh >= 0.05, sorted by test rejected p>=.2:")
print(t.filter((pl.col("v_n") >= 300) & (((pl.col("true_noH") - pl.col("true_all")) >= 0.02) | (pl.col("H_sh") >= 0.05)))
       .sort("t_rej_p.2", descending=True).head(40))
print("EMPTY groups:")
print(t.filter(pl.col("g").str.starts_with("EMPTY")).sort("t_n", descending=True).head(25))
v = pl.read_parquet(OUT / "val_feats.parquet").filter(pl.col("s1_val"))
tt = pl.read_parquet(OUT / "test_feats.parquet")
for nm, d in (("val", v), ("test", tt)):
    x = d.filter(pl.col("prob").is_between(0.05, 0.95))
    agg = [pl.len().alias("n"), pl.col("prob").mean().round(3).alias("mprob"), pl.col("acc").mean().round(3).alias("acc")]
    if nm == "val":
        agg += [(pl.col("cls") == "H").mean().round(3).alias("H_sh"), (pl.col("cls") == "R").mean().round(3).alias("R_sh"),
                pl.col("y").mean().round(3).alias("true_all"), pl.col("y").filter(pl.col("cls") != "H").mean().round(3).alias("true_noH")]
    print(nm, "uncertain (.05-.95) by country x empty address:")
    print(x.group_by("q_country", "q_addr_empty").agg(agg).sort("q_country", "q_addr_empty"))
# val H share and true rates in the uncertain zone by prob bucket (calibration with / without H)
x = v.with_columns((pl.col("prob") * 10).floor().clip(0, 9).alias("b"))
print("val calibration by prob decile, EMPTY vs addr:")
print(x.group_by("q_addr_empty", "b").agg(pl.len().alias("n"), pl.col("prob").mean().round(3).alias("mprob"),
      (pl.col("cls") == "H").mean().round(3).alias("H_sh"), pl.col("y").mean().round(3).alias("true_all"),
      pl.col("y").filter(pl.col("cls") != "H").mean().round(3).alias("true_noH")).sort("q_addr_empty", "b"))

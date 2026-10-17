"""Agent FR a2: marker table per class from a1 output.
uv run --no-project --with polars==1.44.2 python scripts/agent_fr/a2_markers.py
"""
import polars as pl

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(250)
SCR = "local_data/work"
x = pl.read_parquet(f"{SCR}/a1_pairs.parquet").filter(pl.col("country") == "France")
slots = pl.read_parquet(f"{SCR}/a1_slots.parquet").filter(pl.col("country") == "France")
NS1 = 259452
full = ((pl.col("src") == 2) & (pl.col("n_other") >= 5)) | ((pl.col("src") == 3) & (pl.col("n_other") >= 6))
low = pl.col("n_other").is_between(2, 3)
den_full = slots.filter(full)["n_slots"].sum(); den_low = slots.filter(low)["n_slots"].sum()
print("slots full", den_full, "low", den_low)
agg = [pl.len().alias("n"), (pl.len() / NS1 * 1000).round(2).alias("per1k"), pl.col("brk_new").mean().mul(100).round(2).alias("brk%"),
       pl.col("brk_any").mean().mul(100).round(2).alias("brkany%"), (pl.col("src") == 2).mean().round(3).alias("s2"),
       (full.sum() / den_full * 1e3).alias("r_full"), (low.sum() / den_low * 1e3).alias("r_low")]
for st in ("acc", "una"):
    t = x.filter(pl.col("status") == st).group_by("nrel", "hrel", "srel").agg(agg).with_columns((pl.col("r_full") / pl.col("r_low")).round(3).alias("cap"))
    t = t.drop("r_full", "r_low").filter(pl.col("n") >= 300).sort("n", descending=True)
    print(f"\n==== France {st}")
    print(t.head(70))

"""Agent FR a9: size-bias marker (mean other accepted from the same source at the S1) for French accepted classes, split by
single vs multi-tenant S1 address. Wrong-tenant or decoy acceptances are not size-biased (look like unassigned decoys)."""
import re, unicodedata
import polars as pl
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(250)
SCR = "local_data/work"
def norm(s): return unicodedata.normalize("NFKD", s or "").encode("ascii","ignore").decode().lower()
s1 = pl.read_parquet("local_data/test_source1.parquet", columns=["business_address","country"]).with_row_index("s1").with_columns(pl.col("s1").cast(pl.Int64))
s1 = s1.with_columns(pl.Series("akey", [" ".join(sorted(re.findall(r"[a-z0-9]+", norm(a)))) for a in s1["business_address"].to_list()]))
s1 = s1.join(s1.group_by("country", "akey").len().rename({"len": "ten"}), on=["country", "akey"]).select("s1", (pl.col("ten") > 1).alias("multi"))
x = pl.read_parquet(f"{SCR}/a1_pairs.parquet").filter(pl.col("country") == "France").with_columns(pl.col("s1").cast(pl.Int64)).join(s1, on="s1")
t = x.group_by("status", "nrel", "hrel", "multi", "src").agg(pl.len().alias("n"), pl.col("n_other").mean().round(3).alias("m"))
p = t.pivot(on="src", index=["status", "nrel", "hrel", "multi"], values=["n", "m"]).fill_null(0).filter((pl.col("n_2") + pl.col("n_3")) >= 800)
print(p.sort("status", "nrel", "hrel", "multi"))

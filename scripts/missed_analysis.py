"""Which validation true pairs does retrieval miss (v2 run), and why? Run on worker: python scripts/missed_analysis.py /work/run2"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, "src")
from ber.pipeline import labels  # noqa: E402

work = Path(sys.argv[1])
lab = labels(Path("/work/data"), work)
val_q = lab.filter(pl.col("is_val") & pl.col("parent").is_not_null()).select(pl.col("rid").alias("q"), "parent")
got = (pl.scan_parquet(str(work / "train" / "pairs_*.parquet"))
         .join(val_q.lazy(), on="q").filter(pl.col("s1") == pl.col("parent"))
         .select("q", "rank_c", "rank_a", "rank_n", "cos_a", "cos_n").collect())
rec = pl.read_parquet(work / "train" / "records.parquet",
                      columns=["business_name", "business_address", "country", "name_nonascii", "addr_empty", "house",
                               "addr_n", "name_n"]).with_row_index("rid")
t = val_q.join(got, on="q", how="left").with_columns(pl.col("rank_c").is_not_null().alias("found"))
t = t.join(rec.select(pl.col("rid").alias("q"), "country", "name_nonascii", "addr_empty", pl.col("house").alias("q_house")), on="q")
t = t.join(rec.select(pl.col("rid").alias("parent"), pl.col("house").alias("p_house")), on="parent")
t = t.with_columns(((pl.col("q_house") != "") & (pl.col("q_house") == pl.col("p_house"))).alias("house_same"))
print("recall", t["found"].mean(), "missed", (~t["found"]).sum(), "of", t.height)
for c in ("country", "name_nonascii", "addr_empty", "house_same"):
    print(t.group_by(c).agg(pl.len(), pl.col("found").mean().alias("recall"), (~pl.col("found")).sum().alias("missed")).sort(c))
f = t.filter(pl.col("found"))
print("which retriever found them (rank<99):",
      f.select((pl.col("rank_c") < 99).mean().alias("comb"), (pl.col("rank_a") < 99).mean().alias("addr"),
               (pl.col("rank_n") < 99).mean().alias("name"),
               ((pl.col("rank_c") == 99) & (pl.col("rank_a") == 99)).mean().alias("only_name"),
               ((pl.col("rank_c") == 99) & (pl.col("rank_n") == 99)).mean().alias("only_addr")))
m = t.filter(~pl.col("found")).sample(25, seed=3)
m = (m.join(rec.select(pl.col("rid").alias("q"), pl.col("business_name").alias("q_name"), pl.col("business_address").alias("q_addr")), on="q")
      .join(rec.select(pl.col("rid").alias("parent"), pl.col("business_name").alias("p_name"), pl.col("business_address").alias("p_addr")), on="parent"))
with pl.Config(tbl_rows=25, fmt_str_lengths=60, tbl_width_chars=250):
    print(m.select("country", "q_name", "p_name", "q_addr", "p_addr"))

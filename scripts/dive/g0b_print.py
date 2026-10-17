import polars as pl
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_width_chars(400); pl.Config.set_fmt_str_lengths(80)
R = "/work/run4/train/records.parquet"
rec = pl.scan_parquet(R).select("business_name", "business_address", "src", "country").with_row_index("rid")
tp = pl.read_parquet("/work/dive/tp.parquet")
for c in ["US", "India"]:
    s = tp.filter(pl.col("country") == c).sample(40, seed=1)
    r = rec.join(s.lazy().select(pl.col("s1r").alias("rid")).unique(), on="rid").collect()
    q = rec.join(s.lazy().select(pl.col("mr").alias("rid")), on="rid").collect()
    d = s.join(r.select(pl.col("rid").alias("s1r"), pl.col("business_name").alias("n1"), pl.col("business_address").alias("a1")), on="s1r") \
         .join(q.select(pl.col("rid").alias("mr"), pl.col("business_name").alias("n2"), pl.col("business_address").alias("a2")), on="mr")
    for row in d.iter_rows(named=True):
        print(f"S{row['msrc']} | {row['n1']!r:45} -> {row['n2']!r:45} | {row['a1']!r} -> {row['a2']!r}")
    print("-" * 100)

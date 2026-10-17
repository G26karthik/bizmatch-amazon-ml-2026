"""Generator dive step 0: id->rid pair table + raw samples of true pairs and decoys (raw-unmatched S2/S3)."""
import polars as pl
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_width_chars(400); pl.Config.set_fmt_str_lengths(90)
R = "/work/run4/train/records.parquet"
ids = pl.scan_parquet(R).select("entity_id", "src", "country").with_row_index("rid").collect()
print(ids.group_by("src", "country").len().sort("src", "country"))
tp = pl.read_parquet("/work/data/train_pairs.parquet")
m = ids.select(pl.col("entity_id"), pl.col("rid"))
tp = tp.join(m.rename({"entity_id": "s1", "rid": "s1r"}), on="s1").join(m.rename({"entity_id": "m", "rid": "mr"}), on="m")
tp = tp.join(ids.select(pl.col("rid").alias("mr"), pl.col("src").alias("msrc"), pl.col("country")), on="mr")
tp.select("s1r", "mr", "msrc", "country").write_parquet("/work/dive/tp.parquet")
print("true pairs", tp.height)
un = ids.filter(pl.col("src") > 1).join(tp.select(pl.col("mr").alias("rid")), on="rid", how="anti")
un.select("rid", "src", "country").write_parquet("/work/dive/decoys.parquet")
print("raw unmatched", un.height, un.group_by("src", "country").len().sort("src", "country"))

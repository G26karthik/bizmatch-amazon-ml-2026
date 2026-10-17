"""Fetch raw text for a 300k sample of true pairs (both sides) -> /work/dive/tp_s2.parquet (single records scan)."""
import polars as pl
R = "/work/run4/train/records.parquet"
tp = pl.read_parquet("/work/dive/tp.parquet").with_columns(pl.col("s1r").cast(pl.UInt32), pl.col("mr").cast(pl.UInt32)).sample(300000, seed=7)
need = pl.concat([tp["s1r"], tp["mr"]]).unique().to_frame("rid")
rec = pl.scan_parquet(R).select("business_name", "business_address").with_row_index("rid").join(need.lazy(), on="rid").collect()
s = tp.join(rec.rename({"rid": "s1r", "business_name": "business_name_1", "business_address": "business_address_1"}), on="s1r") \
      .join(rec.rename({"rid": "mr", "business_name": "business_name_2", "business_address": "business_address_2"}), on="mr")
s.write_parquet("/work/dive/tp_s2.parquet")
print("tp_s2", s.height)

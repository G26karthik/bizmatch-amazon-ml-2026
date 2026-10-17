"""Tag the cross-encoder training pairs (kaggle/ce3_ds/ce_train.parquet, 1.2M, text only) with test-like flags.
is_h   : record text belongs to a copy of a 19%-hidden S1 (H record; test has none) -> drop for test-like training
is_orph: record is a genuine unmatched record (noise/decoy, not H) -> optional upweight (test has ~2x)
Labels/hiding exactly as scripts/resid_model.py (SEED 42, HIDE 19, hash(SEED+77)).
Out: local_data/fast/tl_train.parquet (text_a, text_b, y, is_h, is_orph)
"""
import polars as pl

SEED, HIDE = 42, 19
rec = pl.concat([pl.read_parquet(f"local_data/train_source{s}.parquet", columns=["entity_id", "business_name", "business_address"])
                 .with_columns(pl.lit(s).cast(pl.Int8).alias("src")) for s in (1, 2, 3)]).with_row_index("rid")
rec = rec.with_columns(((pl.col("src") == 1) & ((pl.col("entity_id").hash(SEED + 77) % 100) < HIDE)).alias("hidden"))
tp = pl.read_parquet("local_data/train_pairs.parquet")
hid = rec.filter(pl.col("hidden")).select(pl.col("entity_id").alias("s1"))
hm = tp.join(hid, on="s1").select(pl.col("m").alias("entity_id"))
live_m = tp.join(hid, on="s1", how="anti").select(pl.col("m").alias("entity_id"))
r23 = rec.filter(pl.col("src") != 1).select(
    "rid", "entity_id",
    (pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null("")).alias("text_b"))
r23 = r23.with_columns(pl.col("entity_id").is_in(hm["entity_id"].implode()).alias("h"),
                       pl.col("entity_id").is_in(live_m["entity_id"].implode()).alias("m"))
r23 = r23.with_columns((~pl.col("h") & ~pl.col("m")).alias("o"))
print("records S2/S3", r23.height, "H", int(r23["h"].sum()), "orphans", int(r23["o"].sum()), flush=True)
agg = r23.group_by("text_b").agg(pl.col("h").any().alias("is_h"), pl.col("o").all().alias("is_orph"), pl.len().alias("n_txt"))
tr = pl.read_parquet("kaggle/ce3_ds/ce_train.parquet").join(agg, on="text_b", how="left")
print("unmatched text_b:", tr["is_h"].null_count(), "of", tr.height, flush=True)
tr = tr.with_columns(pl.col("is_h").fill_null(False), pl.col("is_orph").fill_null(False))
print(tr.group_by("is_h", "is_orph").agg(pl.len(), pl.col("y").mean()).sort("is_h", "is_orph"), flush=True)
tr.select("text_a", "text_b", "y", "is_h", "is_orph").write_parquet("local_data/fast/tl_train.parquet")
print("wrote local_data/fast/tl_train.parquet", tr.height)

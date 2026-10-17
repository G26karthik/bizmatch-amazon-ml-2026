"""uncapped-retrieval probe: population sizes (train: records with no pruned candidate; test: not in fr15 candidates)."""
import polars as pl
SEED, HIDE = 42, 19
rec = pl.concat([pl.read_parquet(f"local_data/train_source{s}.parquet", columns=["entity_id", "business_address", "country"]).with_columns(pl.lit(s).cast(pl.Int8).alias("src")) for s in (1, 2, 3)]).with_row_index("rid")
rec = rec.with_columns(((pl.col("src") == 1) & ((pl.col("entity_id").hash(SEED + 77) % 100) < HIDE)).alias("hidden"))
pp = pl.read_parquet("local_data/pruned_pairs_train.parquet")
print("pruned pairs", pp.height, "distinct q", pp["q"].n_unique(), "distinct s1", pp["s1"].n_unique())
h = rec.filter(pl.col("hidden")).select(pl.col("rid").cast(pl.UInt32).alias("s1"))
print("pruned pairs to hidden S1s:", pp.join(h, on="s1").height, " hidden S1s:", h.height)
q = pp.select(pl.col("q").unique()).with_columns(pl.lit(True).alias("inpp"))
r = rec.filter(pl.col("src") != 1).with_columns(pl.col("rid").cast(pl.UInt32)).join(q, left_on="rid", right_on="q", how="left").with_columns(pl.col("inpp").fill_null(False))
r = r.with_columns((pl.col("business_address").str.strip_chars() == "").alias("emp"))
print(r.group_by("country", "emp").agg(pl.len(), pl.col("inpp").mean().round(4)).sort("country", "emp"))
# test
cp = pl.read_csv("output/v28q_fr15/candidate_pairs.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
cp = cp.with_columns(pl.col("candidate_entity_ids").str.split(",")).explode("candidate_entity_ids").filter(pl.col("candidate_entity_ids") != "")
print("test candidate pairs", cp.height, "distinct records", cp["candidate_entity_ids"].n_unique())
t = pl.concat([pl.read_parquet(f"local_data/test_source{s}.parquet", columns=["entity_id", "business_address", "country"]) for s in (2, 3)])
t = t.join(cp.select(pl.col("candidate_entity_ids").unique().alias("entity_id")).with_columns(pl.lit(True).alias("incp")), on="entity_id", how="left").with_columns(pl.col("incp").fill_null(False))
t = t.with_columns((pl.col("business_address").str.strip_chars() == "").alias("emp"))
print(t.group_by("country", "emp").agg(pl.len(), pl.col("incp").mean().round(4)).sort("country", "emp"))

"""Select the new-evidence additions. The proposal pool (v28q_fr45) is compared with v28q_fr22 and its added pairs are
split by whether the record had an uncertain-zone best pair (already scored by the calibrated model, so a test-like
probability pn exists) and by country and source step, with the expected F0.5 effect per group using pn as precision
for in-zone pairs. Writes the additions for records the calibrated model had not scored
(local_data/work/never_scored_adds.parquet), which v28q_fr48 adds to v28q_fr22.

uv run --no-project --with polars==1.44.2 python scripts/deep/d42_postmortem.py
"""
import polars as pl

pl.Config.set_tbl_rows(-1)


def pairs(v):
    g = pl.read_csv(f"output/{v}/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
    return g.select(pl.col("source1_entity_id").alias("s1e"), pl.col("matched_entity_ids").fill_null("").str.split(",").alias("m")).explode("m").filter(pl.col("m") != "")


a22, a45 = pairs("v28q_fr22"), pairs("v28q_fr45")
add = a45.join(a22, on=["s1e", "m"], how="anti")
rem = a22.join(a45, on=["s1e", "m"], how="anti")
print("fr45 vs fr22: added", add.height, "removed", rem.height)
rec = pl.concat([pl.read_parquet(f"local_data/test_source{s}.parquet", columns=["entity_id", "country", "business_address"]) for s in (1, 2, 3)]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
rid = rec.select(pl.col("entity_id"), "rid", "country", (pl.col("business_address").fill_null("").str.strip_chars() == "").alias("empty"))
add = add.join(rid.rename({"entity_id": "m", "rid": "q"}), on="m").join(rid.select(pl.col("entity_id").alias("s1e"), pl.col("rid").alias("s1")), on="s1e")
unc = pl.read_parquet("local_data/llm/unc_test.parquet").select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32).alias("best"), "prob")
pn = pl.read_parquet("local_data/llm/resid_test_tl_v34.parquet").select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32).alias("best"), "pn")
add = add.join(unc, on="q", how="left").join(pn, on=["q", "best"], how="left")
add = add.with_columns(pl.when(pl.col("best").is_null()).then(pl.lit("never_scored")).when(pl.col("best") == pl.col("s1")).then(pl.lit("in_zone_same_S1")).otherwise(pl.lit("in_zone_other_S1")).alias("zone"))
print(add.group_by("country", "empty", "zone").agg(pl.len(), pl.col("pn").mean().round(3).alias("mean_pn"), pl.col("prob").mean().round(3).alias("mean_prob")).sort("country", "empty", "zone"))
z = add.filter(pl.col("zone") == "in_zone_same_S1")
eg = (z["pn"] * 0.065 - (1 - z["pn"]) * 0.21).sum()
print(f"in-zone same-S1 adds: {z.height}, mean pn {z['pn'].mean():.3f}, expected S1-units using pn: {eg:.1f} (~{eg * 5.6e-7:+.6f} LB)")
add.select("m", "s1e", "country", "empty", "zone", "pn").write_parquet("local_data/work/postmortem_adds.parquet")
# additions without counter-evidence (the model never scored the record) -> input of v28q_fr48
ns = add.filter(pl.col("zone") == "never_scored").select("m", pl.col("s1e").alias("s1"), "country",
        pl.when(pl.col("empty")).then(pl.lit("never_scored_empty")).otherwise(pl.lit("never_scored_addr")).alias("kind"))
ns.write_parquet("local_data/work/never_scored_adds.parquet")
print("never-scored additions written:", ns.height)

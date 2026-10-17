"""blindA: test S3 rows exactly equal to an S1 (name+addr) - what are they?"""
import polars as pl
O = "/work/agent_blindA"
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(80)
t = {k: pl.read_parquet(f"{O}/raw_test_s{k}.parquet").with_columns(pl.col("name").fill_null(""), pl.col("addr").fill_null("")) for k in [1, 2, 3]}
s = {k: pl.read_parquet(f"{O}/raw_train_s{k}.parquet").with_columns(pl.col("name").fill_null(""), pl.col("addr").fill_null("")) for k in [1, 2, 3]}
sub = pl.read_csv("/work/out_fr14/matching_results.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
sp = sub.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1_sub", "matched_entity_ids": "entity_id"})
for nm, d in [("test", t), ("train", s)]:
    for k in [2, 3]:
        j = d[k].join(d[1].select("name", "addr", pl.col("entity_id").alias("s1"), pl.col("country").alias("c1")), on=["name", "addr"])
        print(nm, f"S{k}", "exact name+addr = S1:", j.height, j.group_by("country").len().to_dicts())
        if nm == "test":
            j = j.join(sp, on="entity_id", how="left")
            print("  assigned to that S1:", (j["s1_sub"] == j["s1"]).sum(), " to other:", ((j["s1_sub"] != j["s1"]) & j["s1_sub"].is_not_null()).sum(), " unassigned:", j["s1_sub"].is_null().sum())
            print(j.head(15).select("entity_id", "name", "addr", "country", "s1", "s1_sub"))
        # exact name only (any address) rates by country
        jn = d[k].join(d[1].select("name", pl.col("country")).unique(), on=["name", "country"], how="semi")
        print(f"  {nm} S{k} exact name (same country) as some S1: frac", round(jn.height / d[k].height, 4), jn.group_by("country").len().sort("country").to_dicts())
        ja = d[k].filter(pl.col("addr") != "").join(d[1].select("addr", "country").unique(), on=["addr", "country"], how="semi")
        print(f"  {nm} S{k} exact addr as some S1: frac", round(ja.height / d[k].height, 4), ja.group_by("country").len().sort("country").to_dicts())

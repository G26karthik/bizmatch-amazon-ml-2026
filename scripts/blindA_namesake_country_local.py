import polars as pl, sys
R = sys.argv[1]; SUB = sys.argv[2]
norm = lambda c: pl.col(c).str.to_lowercase().str.replace_all(r"[^\p{L}\p{N} ]", " ").str.replace_all(r"\s+", " ").str.strip_chars()
def rd(f):
    return pl.read_csv(f"{R}/{f}", separator="\t", quote_char=None, infer_schema=False).fill_null("")
for split in ["train", "test"]:
    s1 = rd(f"{split}_source1.tsv").with_columns(norm("business_name").alias("nn"))
    c = s1.group_by(["nn", "country"]).len().rename({"len": "ns"})
    s1 = s1.join(c, on=["nn", "country"])
    print(split, "S1 share of names shared with another S1 (normalized):", s1.group_by("country").agg((pl.col("ns") > 1).mean().round(4), pl.col("ns").mean().round(2).alias("mean_group")).sort("country").to_dicts())
    emp = pl.concat([rd(f"{split}_source{k}.tsv").filter(pl.col("business_address") == "") for k in [2, 3]]).with_columns(norm("business_name").alias("nn"))
    e = emp.join(c, on=["nn", "country"], how="left").with_columns(pl.col("ns").fill_null(0))
    print(split, " empty-address S2/S3 records: exact-name S1 count distribution:", e.group_by("country").agg(pl.len(), (pl.col("ns") == 0).mean().round(3).alias("none"), (pl.col("ns") == 1).mean().round(3).alias("unique"), (pl.col("ns") > 1).mean().round(3).alias("ambig")).sort("country").to_dicts())
    if split == "test":
        sub = pl.read_csv(SUB, separator="\t", quote_char=None, infer_schema=False).fill_null("")
        sp = sub.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "entity_id"})
        e = e.join(sp, on="entity_id", how="left")
        print("  test empty-address records assigned rate by country & ns class:", e.with_columns(pl.col("ns").clip(0, 3)).group_by(["country", "ns"]).agg(pl.len(), pl.col("s1").is_not_null().mean().round(3).alias("assigned")).sort(["country", "ns"]).to_dicts())

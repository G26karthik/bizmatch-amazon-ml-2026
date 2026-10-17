import polars as pl, sys
C = sys.argv[1]; R = sys.argv[2]
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(60)
norm = lambda c: pl.col(c).str.to_lowercase().str.replace_all(r"[^\p{L}\p{N} ]", " ").str.replace_all(r"\s+", " ").str.strip_chars()
for split in ["train", "test"]:
    dec = pl.read_parquet(f"{C}/{split}_US_decrec.parquet")
    dec = dec.with_columns(norm("name").str.split(" ").list.set_difference(norm("s1_name").str.split(" ")).alias("addw"))
    cs = dec.group_by(["s1", "h"]).agg(pl.len().alias("cs"), pl.col("addw"), pl.col("name"), pl.col("s1_name").first(), pl.col("d").first(), pl.col("entity_id"))
    two = cs.filter(pl.col("cs") == 2).with_columns(pl.col("addw").list.get(0).list.set_intersection(pl.col("addw").list.get(1)).list.len().alias("shared"))
    three = cs.filter(pl.col("cs") == 3)
    print(f"\n=== {split}: 2-clusters {two.height}, share added word {(two['shared']>0).mean():.3f}; 3-clusters {three.height}")
    if split == "train":
        gt = pl.read_csv(f"{R}/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
        single = gt.filter(pl.col("matched_entity_ids") == "").select(pl.col("source1_entity_id").alias("s1"), pl.lit(True).alias("single"))
        two = two.join(single, on="s1", how="left").with_columns(pl.col("single").fill_null(False))
        print(two.group_by(["single", pl.col("shared") > 0]).len())
        print(two.filter(pl.col("single")).head(12).select("s1_name", "d", "name", "entity_id"))
        print(two.filter(~pl.col("single")).head(8).select("s1_name", "d", "name", "entity_id"))
    else:
        sub = pl.read_csv("../output/v28q_fr14/matching_results.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
        emp = sub.filter(pl.col("matched_entity_ids") == "").select(pl.col("source1_entity_id").alias("s1"), pl.lit(True).alias("pred_empty"))
        two = two.join(emp, on="s1", how="left").with_columns(pl.col("pred_empty").fill_null(False))
        print(two.group_by(["pred_empty", pl.col("shared") > 0]).len().sort(["pred_empty", "shared"]))
        print(two.filter(pl.col("shared") == 0).head(12).select("s1_name", "d", "name", "entity_id", "pred_empty"))
        print(two.filter(pl.col("shared") > 0).head(8).select("s1_name", "d", "name", "entity_id", "pred_empty"))
        print(three.head(6).select("s1_name", "d", "name"))

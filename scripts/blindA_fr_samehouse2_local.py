import polars as pl, sys
C = sys.argv[1]; SUB = sys.argv[2]
pl.Config.set_tbl_rows(70); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(50)
norm = lambda c: pl.col(c).str.to_lowercase().str.replace_all(r"[^\p{L}\p{N} ]", " ").str.replace_all(r"\s+", " ").str.strip_chars()
sub = pl.read_csv(SUB, separator="\t", quote_char=None, infer_schema=False).fill_null("")
sp = sub.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "sub_s1", "matched_entity_ids": "entity_id"})
s1 = pl.read_parquet(f"{C}/test_FR_s1.parquet").select(pl.col("entity_id").alias("s1"), pl.col("name").alias("s1_name"), pl.col("addr").alias("s1_addr"), "K", "h")
kh = s1.group_by(["K", "h"]).len().rename({"len": "nKH"})
print("France S1 (K,h) multiplicity:", kh.group_by(pl.col("nKH").clip(1, 4)).agg(pl.col("nKH").sum().alias("s1s")).sort("nKH").to_dicts())
s1u = s1.join(kh, on=["K", "h"]).filter(pl.col("nKH") == 1)
recs = pl.read_parquet(f"{C}/test_FR_recs.parquet").select("entity_id", "name", "addr", "K", "h").filter(pl.col("h").is_not_null())
j = recs.join(s1u, on=["K", "h"]).join(sp, on="entity_id", how="left")
j = j.with_columns((norm("name").str.split(" ").list.set_intersection(norm("s1_name").str.split(" ")).list.len() / norm("s1_name").str.split(" ").list.unique().list.len()).alias("cov"),
                   pl.when(pl.col("sub_s1") == pl.col("s1")).then(pl.lit("this")).when(pl.col("sub_s1").is_null()).then(pl.lit("none")).otherwise(pl.lit("other")).alias("asg"))
print(j.group_by(["asg", (pl.col("cov") * 4).floor().clip(0, 4).alias("cov4")]).len().sort(["asg", "cov4"]).to_dicts())
x = j.filter((pl.col("asg") == "none")).sample(40, seed=3)
print(x.select("s1_name", "name", "s1_addr", "addr", "cov"))
# where are unassigned same-house records assigned? none by construction; how many S1 at the same street (K) regardless of house?

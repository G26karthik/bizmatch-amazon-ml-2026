"""blindA (local, test): predicted-empty S1s - do they have unassigned same-house records (possible missed copies)?
France (crude French key, cached by blindA_fr_cluster_local.py) vs US test (S1 alone on key) vs US train singletons."""
import polars as pl, sys
C = sys.argv[1]; SUB = sys.argv[2]
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(60)
norm = lambda c: pl.col(c).str.to_lowercase().str.replace_all(r"[^\p{L}\p{N} ]", " ").str.replace_all(r"\s+", " ").str.strip_chars()
sub = pl.read_csv(SUB, separator="\t", quote_char=None, infer_schema=False).fill_null("")
sp = sub.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "sub_s1", "matched_entity_ids": "entity_id"})
ks = sp.group_by("sub_s1").len().rename({"sub_s1": "s1", "len": "k_sub"})
s1 = pl.read_parquet(f"{C}/test_FR_s1.parquet").select(pl.col("entity_id").alias("s1"), pl.col("name").alias("s1_name"), "K", "h", "nK")
# S1s sharing (K,h) with another S1 are ambiguous: drop them
kh = s1.group_by(["K", "h"]).len().rename({"len": "nKH"})
s1 = s1.join(kh, on=["K", "h"]).filter(pl.col("nKH") == 1).join(ks, on="s1", how="left").with_columns(pl.col("k_sub").fill_null(0))
recs = pl.read_parquet(f"{C}/test_FR_recs.parquet").select("entity_id", "name", "addr", "K", "h").filter(pl.col("h").is_not_null())
j = recs.join(s1, on=["K", "h"]).join(sp, on="entity_id", how="left")
j = j.with_columns(norm("name").alias("rn"), norm("s1_name").alias("sn"))
j = j.with_columns((pl.col("rn").str.split(" ").list.set_intersection(pl.col("sn").str.split(" ")).list.len() / pl.col("sn").str.split(" ").list.unique().list.len()).alias("cov"),
                   pl.when(pl.col("sub_s1") == pl.col("s1")).then(pl.lit("this")).when(pl.col("sub_s1").is_null()).then(pl.lit("none")).otherwise(pl.lit("other")).alias("asg"))
print("France S1 unique on (key,house):", s1.height, " pred-empty rate:", round(float((s1["k_sub"] == 0).mean()), 4))
print(" same-house records by assignment:", j.group_by("asg").agg(pl.len(), pl.col("cov").mean().round(3)).to_dicts())
un = j.filter(pl.col("asg") == "none").group_by("s1").agg(pl.len().alias("n_un"), pl.col("cov").max().alias("best_cov"))
x = s1.join(un, on="s1", how="left").with_columns(pl.col("n_un").fill_null(0), pl.col("best_cov").fill_null(0))
print(" P(>=1 unassigned same-house record with cov>=0.5) by k_sub:", x.group_by(pl.col("k_sub").clip(0, 6)).agg(pl.len(), ((pl.col("n_un") > 0) & (pl.col("best_cov") >= 0.5)).mean().round(4).alias("p_cov50"), (pl.col("n_un") > 0).mean().round(4).alias("p_any")).sort("k_sub").to_dicts())
ex = x.filter((pl.col("k_sub") == 0) & (pl.col("best_cov") >= 0.5))
print(" pred-empty France S1s with an unassigned same-house record cov>=0.5:", ex.height)
e = j.join(ex.select("s1"), on="s1").filter(pl.col("asg") == "none").sort("cov", descending=True)
print(e.select("s1_name", "name", "addr", "cov").head(40))
e.write_parquet(f"{C}/fr_pred_empty_samehouse.parquet")

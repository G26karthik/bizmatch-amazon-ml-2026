"""blindA (local): words added by decoys (US offset decoys) that never appear as added words in true copies (all train pairs);
how many submitted test pairs carry such an added word?"""
import polars as pl, sys, gc
R = sys.argv[1]; C = sys.argv[2]; SUB = sys.argv[3]
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(240); pl.Config.set_fmt_str_lengths(60)
norm = lambda c: pl.col(c).str.to_lowercase().str.replace_all(r"[^\p{L}\p{N} ]", " ").str.replace_all(r"\s+", " ").str.strip_chars()
def names(split, k):
    return pl.read_csv(f"{R}/{split}_source{k}.tsv", separator="\t", quote_char=None, infer_schema=False, columns=["entity_id", "business_name", "country"]).fill_null("")
def added(pairs, split):
    s1 = names(split, 1).select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("sn"), "country")
    m = pl.concat([names(split, 2), names(split, 3)]).select(pl.col("entity_id").alias("m"), pl.col("business_name").alias("mn"))
    p = pairs.join(s1, on="s1").join(m, on="m")
    p = p.with_columns(norm("mn").str.split(" ").list.set_difference(norm("sn").str.split(" ")).alias("addw"))
    return p
gt = pl.read_csv(f"{R}/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
pairs = gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "m"})
tp = added(pairs, "train")
cw = tp.select("country", "addw").explode("addw").drop_nulls().group_by(["country", "addw"]).len().rename({"len": "n_copy"})
ncop = tp.group_by("country").len().rename({"len": "N"})
del gt, pairs; gc.collect()
off = pl.read_parquet(f"{C}/us_train_offset_recs.parquet")
dw = off.filter(pl.col("grp") == "nonsingle").select("addw").explode("addw").drop_nulls().group_by("addw").len().rename({"len": "n_decoy"}).sort("n_decoy", descending=True)
nd = off.filter(pl.col("grp") == "nonsingle").height
dw = dw.join(cw.filter(pl.col("country") == "US").select("addw", "n_copy"), on="addw", how="left").with_columns(pl.col("n_copy").fill_null(0))
dw = dw.with_columns((pl.col("n_decoy") / nd * 100).round(3).alias("decoy%"), (pl.col("n_copy") / ncop.filter(pl.col("country") == "US")["N"][0] * 100).round(4).alias("copy%"))
print(dw.head(60))
D = dw.filter((pl.col("n_decoy") >= 50) & (pl.col("n_copy") <= 2))["addw"].to_list()
print("decoy-only words (US decoys >=50, US true copies <=2):", len(D), D)
print("train true pairs carrying such added word by country:", tp.with_columns(pl.col("addw").list.eval(pl.element().is_in(D)).list.any().alias("hit")).group_by("country").agg(pl.col("hit").sum(), pl.len()).to_dicts())
print("India copy counts of those words:", cw.filter((pl.col("country") == "India") & pl.col("addw").is_in(D)).sort("n_copy", descending=True).head(20).to_dicts())
del tp; gc.collect()
sub = pl.read_csv(SUB, separator="\t", quote_char=None, infer_schema=False).fill_null("")
sp = sub.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "m"})
sp = added(sp, "test")
sp = sp.with_columns(pl.col("addw").list.eval(pl.element().filter(pl.element().is_in(D))).alias("hitw"))
hit = sp.filter(pl.col("hitw").list.len() > 0)
print("\nsubmitted pairs with a decoy-only added word:", hit.group_by("country").len().to_dicts(), "of", sp.group_by("country").len().to_dicts())
print(hit.select("country", "hitw").explode("hitw").group_by(["country", "hitw"]).len().sort("len", descending=True).head(30).to_dicts())
print(hit.filter(pl.col("country") != "France").select("s1", "sn", "mn", "hitw").head(40))
hit.select("s1", "m", "country", "sn", "mn", "hitw").write_parquet(f"{C}/sub_decoyword_hits.parquet")

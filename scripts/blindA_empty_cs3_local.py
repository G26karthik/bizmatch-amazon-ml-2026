"""blindA (local, US test): are predicted-empty S1s with 3-record decoy clusters true singletons or missed copies?
For S1s alone on their street key: same-house records (d=0) and whether they are assigned; name similarity.
Train analog with truth for calibration."""
import polars as pl, sys, os
import blindA_lib_local as L
R = sys.argv[1]; C = sys.argv[2]; SUB = sys.argv[3]
L.R = R
pl.Config.set_tbl_rows(50); pl.Config.set_tbl_width_chars(240); pl.Config.set_fmt_str_lengths(55)
norm = lambda c: pl.col(c).str.to_lowercase().str.replace_all(r"[^\p{L}\p{N} ]", " ").str.replace_all(r"\s+", " ").str.strip_chars()

def same_house(split):
    cache = f"{C}/{split}_US_samehouse.parquet"
    if os.path.exists(cache):
        return pl.read_parquet(cache)
    s1 = L.addr_feats(L.scan(f"{split}_source1.tsv").filter(pl.col("country") == "US").collect())
    kc = s1.group_by("K").len().rename({"len": "nK"})
    s1 = s1.join(kc, on="K").filter((pl.col("nK") == 1) & (pl.col("K") != "") & pl.col("h").is_not_null())
    recs = pl.concat([L.addr_feats(L.scan(f"{split}_source{j}.tsv").filter(pl.col("country") == "US").collect()) for j in [2, 3]])
    j = recs.join(s1.select(pl.col("entity_id").alias("s1"), pl.col("name").alias("s1_name"), "K", pl.col("h").alias("h1")), on="K")
    j = j.filter((pl.col("h") == pl.col("h1")) | pl.col("h").is_null())
    j = j.with_columns(norm("name").alias("rn"), norm("s1_name").alias("sn"))
    j = j.with_columns((pl.col("rn") == pl.col("sn")).alias("name_eq"),
                       (pl.col("rn").str.split(" ").list.set_intersection(pl.col("sn").str.split(" ")).list.len() /
                        pl.col("sn").str.split(" ").list.unique().list.len()).alias("s1_tok_cov"),
                       pl.col("h").is_null().alias("no_house"))
    out = j.select("entity_id", "s1", "name", "s1_name", "addr", "name_eq", "s1_tok_cov", "no_house")
    out.write_parquet(cache)
    return out

# ---------- train calibration
tr = same_house("train")
gt = pl.read_csv(f"{R}/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
pairs = gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "entity_id"})
tr = tr.join(pairs.rename({"s1": "true_s1"}), on="entity_id", how="left")
tr = tr.with_columns(pl.when(pl.col("true_s1") == pl.col("s1")).then(pl.lit("this")).when(pl.col("true_s1").is_null()).then(pl.lit("none")).otherwise(pl.lit("other")).alias("truth"))
trh = tr.filter(~pl.col("no_house"))
print("TRAIN same-house records (house present) by truth:", trh.group_by("truth").agg(pl.len(), pl.col("name_eq").mean().round(3), pl.col("s1_tok_cov").mean().round(3)).to_dicts())
print(" P(this S1 | same house, coverage bin):", trh.with_columns((pl.col("s1_tok_cov") * 4).floor().clip(0, 4).alias("cov")).group_by("cov").agg(pl.len(), (pl.col("truth") == "this").mean().round(4).alias("p_true")).sort("cov").to_dicts())
sig = pl.read_parquet(f"{C}/train_US_sig.parquet")
k = pairs.group_by("s1").len().rename({"len": "k"})
sig = sig.join(k, on="s1", how="left").with_columns(pl.col("k").fill_null(0))
unm = trh.filter(pl.col("truth") == "none").group_by("s1").agg(pl.len().alias("n_unm_same"), pl.col("s1_tok_cov").max().alias("best_cov"))
sig = sig.join(unm, on="s1", how="left").with_columns(pl.col("n_unm_same").fill_null(0))
print(" train singletons: P(>=1 unmatched same-house record with cov>=0.5) by max_csize:",
      sig.filter(pl.col("k") == 0).group_by(pl.col("max_csize").clip(0, 3)).agg(pl.len(), ((pl.col("n_unm_same") > 0) & (pl.col("best_cov") >= 0.5)).mean().round(4).alias("p")).sort("max_csize").to_dicts())

# ---------- test
te = same_house("test")
sub = pl.read_csv(SUB, separator="\t", quote_char=None, infer_schema=False).fill_null("")
sp = sub.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "sub_s1", "matched_entity_ids": "entity_id"})
te = te.join(sp, on="entity_id", how="left")
te = te.with_columns(pl.when(pl.col("sub_s1") == pl.col("s1")).then(pl.lit("this")).when(pl.col("sub_s1").is_null()).then(pl.lit("none")).otherwise(pl.lit("other")).alias("asg"))
teh = te.filter(~pl.col("no_house"))
print("\nTEST same-house records (house present) by assignment:", teh.group_by("asg").agg(pl.len(), pl.col("name_eq").mean().round(3), pl.col("s1_tok_cov").mean().round(3)).to_dicts())
print(" unassigned same-house records by coverage bin:", teh.filter(pl.col("asg") == "none").with_columns((pl.col("s1_tok_cov") * 4).floor().clip(0, 4).alias("cov")).group_by("cov").len().sort("cov").to_dicts())
tsig = pl.read_parquet(f"{C}/test_US_sig_sub.parquet")
unm = teh.filter(pl.col("asg") == "none").group_by("s1").agg(pl.len().alias("n_unm_same"), pl.col("s1_tok_cov").max().alias("best_cov"))
tsig = tsig.join(unm, on="s1", how="left").with_columns(pl.col("n_unm_same").fill_null(0))
print(" test predicted-empty S1s: P(>=1 unassigned same-house record with cov>=0.5) by max_csize:",
      tsig.filter(pl.col("k_sub") == 0).group_by(pl.col("max_csize").clip(0, 3)).agg(pl.len(), ((pl.col("n_unm_same") > 0) & (pl.col("best_cov") >= 0.5)).mean().round(4).alias("p")).sort("max_csize").to_dicts())
ex = tsig.filter((pl.col("k_sub") == 0) & (pl.col("max_csize") >= 3)).head(400)
exr = teh.join(ex.select("s1"), on="s1").filter(pl.col("asg") == "none").sort("s1").head(30)
print(exr.select("s1_name", "name", "addr", "s1_tok_cov"))
dec = pl.read_parquet(f"{C}/test_US_decrec.parquet").join(ex.select("s1").head(8), on="s1").sort(["s1", "h"])
print(dec.select("s1", "s1_name", "d", "name"))

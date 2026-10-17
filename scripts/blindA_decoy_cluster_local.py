"""blindA (local, US): truth-free decoy-cluster signature per S1, train vs test.
For S1s alone on their (street,city) key: records on the same key whose house = S1 house + {1,2,3,4,5,7,9,11,13,21}
and whose name adds a word to the S1 name ("decoy-offset records"); cluster them by house.
Train: P(singleton | signature). Test: prevalence + what the submission does with those S1s.
"""
import polars as pl, sys, os
import blindA_lib_local as L
R = sys.argv[1]; C = sys.argv[2]; SUB = sys.argv[3]
L.R = R
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(230)
OFF = [1, 2, 3, 4, 5, 7, 9, 11, 13, 21]
norm = lambda c: pl.col(c).str.to_lowercase().str.replace_all(r"[^\p{L}\p{N} ]", " ").str.replace_all(r"\s+", " ").str.strip_chars()

def signature(split, country="US"):
    cache = f"{C}/{split}_{country}_sig.parquet"
    if os.path.exists(cache):
        return pl.read_parquet(cache), pl.read_parquet(cache.replace("_sig", "_decrec"))
    s1 = L.addr_feats(L.scan(f"{split}_source1.tsv").filter(pl.col("country") == country).collect())
    kc = s1.group_by("K").len().rename({"len": "nK"})
    s1 = s1.join(kc, on="K").filter((pl.col("nK") == 1) & (pl.col("K") != "") & pl.col("h").is_not_null())
    recs = pl.concat([L.addr_feats(L.scan(f"{split}_source{j}.tsv").filter(pl.col("country") == country).collect()) for j in [2, 3]])
    j = recs.join(s1.select(pl.col("entity_id").alias("s1"), pl.col("name").alias("s1_name"), "K", pl.col("h").alias("h1")), on="K")
    j = j.with_columns((pl.col("h") - pl.col("h1")).alias("d"), norm("name").alias("rn"), norm("s1_name").alias("sn"))
    j = j.with_columns(pl.col("rn").str.split(" ").list.set_difference(pl.col("sn").str.split(" ")).list.len().alias("n_add"),
                       pl.col("sn").str.split(" ").list.set_difference(pl.col("rn").str.split(" ")).list.len().alias("n_drop"))
    dec = j.filter(pl.col("d").is_in(OFF) & (pl.col("n_add") > 0)).select("entity_id", "s1", "h", "d", "name", "s1_name", "n_add", "n_drop")
    cl = dec.group_by(["s1", "h"]).len().rename({"len": "csize"})
    per = cl.group_by("s1").agg(pl.len().alias("n_clusters"), pl.col("csize").max().alias("max_csize"), pl.col("csize").sum().alias("n_decrec"))
    out = s1.select(pl.col("entity_id").alias("s1")).join(per, on="s1", how="left").with_columns(pl.all().exclude("s1").fill_null(0))
    out.write_parquet(cache); dec.write_parquet(cache.replace("_sig", "_decrec"))
    return out, dec

# train with truth
tr, trdec = signature("train")
gt = pl.read_csv(f"{R}/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
pairs = gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "m"})
k = pairs.group_by("s1").len().rename({"len": "k"})
tr = tr.join(k, on="s1", how="left").with_columns(pl.col("k").fill_null(0))
print("TRAIN US S1 alone on key:", tr.height, "singleton rate", round((tr["k"] == 0).mean(), 4))
print(tr.with_columns(pl.col("max_csize").clip(0, 3)).group_by(["max_csize"]).agg(pl.len(), (pl.col("k") == 0).mean().round(4).alias("P_single"), pl.col("k").mean().round(3).alias("mean_k")).sort("max_csize"))
print(tr.with_columns(pl.col("n_clusters").clip(0, 3)).group_by(["n_clusters", pl.col("max_csize").clip(0, 3)]).agg(pl.len(), (pl.col("k") == 0).mean().round(4).alias("P_single")).sort(["n_clusters", "max_csize"]))
# are the decoy records themselves matched (truth) ?
td = trdec.join(pairs.select(pl.col("m").alias("entity_id"), pl.col("s1").alias("true_s1")), on="entity_id", how="left")
print("train decoy-offset records: n", td.height, "truly matched to this S1:", (td["true_s1"] == td["s1"]).sum(), "to another S1:", ((td["true_s1"] != td["s1"]) & td["true_s1"].is_not_null()).sum())
# do the two records of a 2-cluster share the added word(s)?
two = trdec.join(trdec.group_by(["s1", "h"]).len().filter(pl.col("len") == 2).select("s1", "h"), on=["s1", "h"])
two = two.with_columns(norm("name").str.split(" ").list.set_difference(norm("s1_name").str.split(" ")).alias("addw"))
g = two.group_by(["s1", "h"]).agg(pl.col("addw").first().alias("a"), pl.col("addw").last().alias("b"))
g = g.with_columns(pl.col("a").list.set_intersection(pl.col("b")).list.len().alias("shared"))
print("2-record clusters:", g.height, "share an added word:", (g["shared"] > 0).mean())

# test
te, tedec = signature("test")
sub = pl.read_csv(SUB, separator="\t", quote_char=None, infer_schema=False).fill_null("")
sp = sub.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "m"})
ks = sp.group_by("s1").len().rename({"len": "k_sub"})
te = te.join(ks, on="s1", how="left").with_columns(pl.col("k_sub").fill_null(0))
print("\nTEST US S1 alone on key:", te.height, "predicted-empty rate", round((te["k_sub"] == 0).mean(), 4))
print(te.with_columns(pl.col("max_csize").clip(0, 3)).group_by(["max_csize"]).agg(pl.len(), (pl.col("k_sub") == 0).mean().round(4).alias("P_pred_empty"), pl.col("k_sub").mean().round(3).alias("mean_k_sub")).sort("max_csize"))
print(te.with_columns(pl.col("n_clusters").clip(0, 3)).group_by(["n_clusters", pl.col("max_csize").clip(0, 3)]).agg(pl.len(), (pl.col("k_sub") == 0).mean().round(4).alias("P_pred_empty")).sort(["n_clusters", "max_csize"]))
tt = tedec.join(sp.select(pl.col("m").alias("entity_id"), pl.col("s1").alias("sub_s1")), on="entity_id", how="left")
print("test decoy-offset records:", tt.height, "assigned to this S1:", (tt["sub_s1"] == tt["s1"]).sum(), "to another:", ((tt["sub_s1"] != tt["s1"]) & tt["sub_s1"].is_not_null()).sum())
te.write_parquet(f"{C}/test_US_sig_sub.parquet")

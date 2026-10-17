"""blindA (local, US train): decoy-offset records per S1 by k: count distribution, source, name relation.
Caches joined tables to scratch parquet for iteration."""
import polars as pl, sys, os
import blindA_lib_local as L
R = sys.argv[1]; C = sys.argv[2]
L.R = R
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(230); pl.Config.set_fmt_str_lengths(60)
norm = lambda c: pl.col(c).str.to_lowercase().str.replace_all(r"[^\p{L}\p{N} ]", " ").str.replace_all(r"\s+", " ").str.strip_chars()
if not os.path.exists(f"{C}/us_join.parquet"):
    s1 = L.addr_feats(L.scan("train_source1.tsv").filter(pl.col("country") == "US").collect())
    gt = pl.read_csv(f"{R}/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
    pairs = gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "m"})
    k = pairs.group_by("s1").len().rename({"len": "k"})
    s1 = s1.join(k, left_on="entity_id", right_on="s1", how="left").with_columns(pl.col("k").fill_null(0))
    s1.write_parquet(f"{C}/us_s1.parquet")
    recs = pl.concat([L.addr_feats(L.scan(f"train_source{j}.tsv").filter(pl.col("country") == "US").collect()) for j in [2, 3]])
    recs = recs.join(pairs.select(pl.col("m").alias("entity_id"), pl.col("s1").alias("true_s1")), on="entity_id", how="left")
    recs.write_parquet(f"{C}/us_recs.parquet")
    j = recs.filter(pl.col("K") != "").join(s1.select(pl.col("entity_id").alias("s1"), pl.col("name").alias("s1_name"), "K", pl.col("h").alias("h1"), "k"), on="K")
    j = j.with_columns((pl.col("h") - pl.col("h1")).alias("d")).filter(pl.col("true_s1").is_null() | (pl.col("true_s1") == pl.col("s1")))
    j.write_parquet(f"{C}/us_join.parquet")
s1 = pl.read_parquet(f"{C}/us_s1.parquet"); j = pl.read_parquet(f"{C}/us_join.parquet")
kc = s1.group_by("K").len().rename({"len": "nK"})
s1 = s1.join(kc, on="K")
OFF = [1, 2, 3, 4, 5, 7, 9, 11, 13, 21]
um = j.filter(pl.col("true_s1").is_null()).join(kc, on="K").filter(pl.col("nK") == 1)  # S1 alone on its street key
um = um.with_columns(norm("name").alias("rn"), norm("s1_name").alias("sn"))
um = um.with_columns(
    (pl.col("rn") == pl.col("sn")).alias("name_eq"),
    pl.col("rn").str.split(" ").list.set_difference(pl.col("sn").str.split(" ")).list.len().alias("n_add"),
    pl.col("sn").str.split(" ").list.set_difference(pl.col("rn").str.split(" ")).list.len().alias("n_drop"),
    pl.col("entity_id").str.slice(0, 2).alias("src"))
dec = um.filter(pl.col("d").is_in(OFF))
dz = um.filter(pl.col("d") == 0)
su = s1.filter(pl.col("nK") == 1)
print("S1 alone on street key:", su.height, "singletons", (su["k"] == 0).sum())
per = dec.group_by("s1").agg(pl.len().alias("ndec"), (pl.col("src") == "S2").sum().alias("ndec2"), pl.col("h").n_unique().alias("ndec_h"))
su = su.join(per, left_on="entity_id", right_on="s1", how="left").with_columns(pl.col("ndec").fill_null(0), pl.col("ndec2").fill_null(0), pl.col("ndec_h").fill_null(0))
print("ndec distribution by k (rows: k; cols ndec 0..4+):")
x = su.with_columns(pl.col("ndec").clip(0, 4).alias("nd"), pl.col("k").clip(0, 6).alias("kc")).group_by(["kc", "nd"]).len().pivot(on="nd", index="kc", values="len").sort("kc")
print(x)
print("distinct decoy houses by k:", su.with_columns(pl.col("ndec_h").clip(0, 3).alias("nh"), pl.col("k").clip(0, 6).alias("kc")).group_by(["kc", "nh"]).len().pivot(on="nh", index="kc", values="len").sort("kc"))
dec = dec.join(su.select(pl.col("entity_id").alias("s1")), on="s1")
dec = dec.with_columns(pl.col("k").clip(0, 3).alias("kc"))
print("decoy name relation by k:", dec.group_by("kc").agg(pl.len(), pl.col("name_eq").mean().round(4), (pl.col("n_add") > 0).mean().round(4).alias("adds_word"),
      ((pl.col("n_add") == 0) & (pl.col("n_drop") > 0)).mean().round(4).alias("drop_only"), (pl.col("src") == "S2").mean().round(4).alias("S2")).sort("kc"))
print("decoy offset by k:", dec.group_by(["kc", "d"]).len().pivot(on="kc", index="d", values="len").sort("d"))
dz = dz.join(su.select(pl.col("entity_id").alias("s1")), on="s1").with_columns(pl.col("k").clip(0, 3).alias("kc"))
print("unmatched at SAME house by k:", dz.group_by("kc").agg(pl.len(), pl.col("name_eq").mean().round(4), (pl.col("n_add") > 0).mean().round(4).alias("adds_word")).sort("kc"))
print("P(>=1 same-house unmatched) by k:", su.join(dz.group_by("s1").len().rename({"len": "nz"}), left_on="entity_id", right_on="s1", how="left").with_columns(pl.col("nz").fill_null(0)).group_by(pl.col("k").clip(0, 6)).agg((pl.col("nz") > 0).mean().round(4), pl.len()).sort("k"))
su.write_parquet(f"{C}/us_su.parquet"); dec.write_parquet(f"{C}/us_dec.parquet")

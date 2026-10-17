"""Model-free check: does test contain copies of businesses absent from S1 (what the hidden-S1 simulation creates),
or only decoys around visible S1s?  Per-S1 densities of record signatures in train raw (no hiding), train with 19%
hidden (pool = visible S1s) and test (pool = all S1s), per country.  Also data-driven decoy words.
"""
import polars as pl, numpy as np
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(40)
SEED = 42
cols = ["entity_id", "src", "country", "name_n", "house", "addr_empty"]


def sig(q, s1pool, tag):
    """q: S2/S3 records (country, name_n, house, addr_empty, cls); s1pool: S1 records of the pool."""
    sn = s1pool.group_by("country", "name_n").agg(pl.len().alias("n_s1_name"))
    snh = s1pool.filter(pl.col("house") != "").group_by("country", "name_n", "house").agg(pl.len().alias("n_s1_nh"))
    qn = q.group_by("country", "name_n").agg(pl.len().alias("n_q_name"))
    qnh = q.filter(pl.col("house") != "").group_by("country", "name_n", "house").agg(pl.len().alias("n_q_nh"))
    d = q.join(sn, on=["country", "name_n"], how="left").join(snh, on=["country", "name_n", "house"], how="left") \
         .join(qn, on=["country", "name_n"], how="left").join(qnh, on=["country", "name_n", "house"], how="left") \
         .with_columns(pl.col("n_s1_name").fill_null(0), pl.col("n_s1_nh").fill_null(0), pl.col("n_q_nh").fill_null(0))
    d = d.with_columns(
        (pl.col("n_s1_name") > 0).alias("name_in_s1"),
        (pl.col("n_s1_name") == 1).alias("name_uniq_s1"),
        (pl.col("n_s1_nh") > 0).alias("nh_in_s1"),
        ((pl.col("n_s1_name") == 0) & (pl.col("n_q_name") >= 2)).alias("orph_cluster_name"),
        ((pl.col("n_s1_nh") == 0) & (pl.col("n_q_nh") >= 2) & (pl.col("n_s1_name") == 0)).alias("orph_cluster_nh"),
        ((pl.col("n_s1_nh") == 0) & (pl.col("n_q_nh") >= 2)).alias("nh_cluster_noS1"),
        (pl.col("addr_empty") & (pl.col("n_s1_name") == 0)).alias("empty_noname"),
        (pl.col("addr_empty") & (pl.col("n_s1_name") == 1)).alias("empty_uniq"),
        (pl.col("addr_empty") & (pl.col("n_s1_name") >= 2)).alias("empty_shared"),
    )
    ns1 = s1pool.group_by("country").agg(pl.len().alias("n_s1"))
    flags = ["name_in_s1", "name_uniq_s1", "nh_in_s1", "orph_cluster_name", "orph_cluster_nh", "nh_cluster_noS1",
             "addr_empty", "empty_noname", "empty_uniq", "empty_shared"]
    out = d.group_by("country").agg(pl.len().alias("recs"), *[pl.col(f).sum() for f in flags]).join(ns1, on="country")
    out = out.select("country", pl.lit(tag).alias("set"), "n_s1", (pl.col("recs") / pl.col("n_s1")).round(3).alias("recs/S1"),
                     *[(pl.col(f) / pl.col("n_s1")).round(4).alias(f) for f in flags]).sort("country")
    return d, out


tr = pl.read_parquet("/work/base/train/records.parquet", columns=cols)
gt = pl.read_parquet("/work/data/train_pairs.parquet")
s1 = tr.filter(pl.col("src") == 1).with_columns(((pl.col("entity_id").hash(SEED + 77) % 100) < 19).alias("hidden"))
q = tr.filter(pl.col("src") != 1).join(gt.rename({"m": "entity_id", "s1": "p_eid"}), on="entity_id", how="left")
q = q.join(s1.select(pl.col("entity_id").alias("p_eid"), pl.col("hidden").alias("p_hidden")), on="p_eid", how="left")
q = q.with_columns(pl.when(pl.col("p_eid").is_null()).then(pl.lit("R")).when(pl.col("p_hidden")).then(pl.lit("H")).otherwise(pl.lit("M")).alias("cls"))
del tr
print("train S2/S3 class counts:", q.group_by("cls").len().sort("cls").rows())
res = []
d_raw, o = sig(q, s1, "train_raw"); res.append(o)
d_vis, o = sig(q, s1.filter(~pl.col("hidden")), "train_hid19"); res.append(o)
# per class breakdown in the hidden setting (rates per visible S1)
nvis = s1.filter(~pl.col("hidden")).group_by("country").agg(pl.len().alias("n_s1"))
flags = ["orph_cluster_name", "orph_cluster_nh", "nh_cluster_noS1", "empty_noname", "empty_uniq", "empty_shared", "addr_empty"]
bc = d_vis.group_by("country", "cls").agg(pl.len().alias("recs"), *[pl.col(f).sum() for f in flags]).join(nvis, on="country")
print("train_hid19 per class, per visible S1:")
print(bc.select("country", "cls", *[(pl.col(f) / pl.col("n_s1")).round(4).alias(f) for f in ["recs"] + flags]).sort("country", "cls"))
bc = d_raw.group_by("country", "cls").agg(pl.len().alias("recs"), *[pl.col(f).sum() for f in flags]).join(
    s1.group_by("country").agg(pl.len().alias("n_s1")), on="country")
print("train_raw per class, per S1:")
print(bc.select("country", "cls", *[(pl.col(f) / pl.col("n_s1")).round(4).alias(f) for f in ["recs"] + flags]).sort("country", "cls"))

# data-driven decoy words: words of S2/S3 names absent from the parent / nearest-name S1, with ~0% match rate (train raw)
w = q.select("country", "cls", pl.col("name_n").str.split(" ").alias("w")).with_row_index("i").explode("w").unique(["i", "w"])
wc = w.group_by("country", "w").agg(pl.len().alias("n"), (pl.col("cls") == "M").mean().alias("rate_M"),
                                    (pl.col("cls") == "R").mean().alias("rate_R"))
s1w = s1.select("country", pl.col("name_n").str.split(" ").alias("w")).with_row_index("i").explode("w").unique(["i", "w"]) \
        .group_by("country", "w").agg(pl.len().alias("n_in_s1"))
wc = wc.join(s1w, on=["country", "w"], how="left").with_columns(pl.col("n_in_s1").fill_null(0))
dec = wc.filter((pl.col("n") >= 2000) & (pl.col("rate_R") >= 0.9)).sort("n", descending=True)
print("decoy-like words (train raw: >=90% of records carrying them are raw orphans):")
print(dec.head(40))
decw = dec.select("country", "w")
del w
te = pl.read_parquet("/work/base/test/records.parquet", columns=cols)
ts1 = te.filter(pl.col("src") == 1)
tq = te.filter(pl.col("src") != 1).with_columns(pl.lit("?").alias("cls"))
del te
d_te, o = sig(tq, ts1, "test"); res.append(o)
print(pl.concat(res).sort("country", "set"))
# decoy-word density per S1: train raw vs test


def dens(qq, s1pool, tag):
    ww = qq.select("country", pl.col("name_n").str.split(" ").alias("w")).with_row_index("i").explode("w").unique(["i", "w"]) \
           .join(decw, on=["country", "w"], how="semi").unique("i")
    return ww.group_by("country").agg(pl.len().alias("dec")).join(s1pool.group_by("country").agg(pl.len().alias("n_s1")), on="country") \
             .with_columns((pl.col("dec") / pl.col("n_s1")).round(4).alias("decoy_word_recs/S1"), pl.lit(tag).alias("set"))


print(pl.concat([dens(q, s1, "train_raw"), dens(tq, ts1, "test")]).sort("country"))
# France: test-only; its decoy words (from france.py) densities
frw = ["participations", "holding", "distribution", "international", "developpement", "groupe", "associes", "fils"]
fr = tq.filter(pl.col("country") == "France")
nfr = ts1.filter(pl.col("country") == "France").height
print("France S1", nfr, "recs/S1", round(fr.height / nfr, 3))
for x in frw:
    n = fr.filter(pl.col("name_n").str.contains(rf"\b{x}\b")).height
    ns = ts1.filter((pl.col("country") == "France") & pl.col("name_n").str.contains(rf"\b{x}\b")).height
    print(f"  {x:15s} recs/S1 {n / nfr:.4f}   S1 names with word /S1 {ns / nfr:.4f}")

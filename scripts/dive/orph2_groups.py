"""Groups of (record, best S1) pairs: validation true rate with vs without hidden-parent orphans (H), the v27 decision,
and the matching test counts.  Also writes per-record feature tables for later scripts.
"""
import json
from pathlib import Path
import numpy as np, polars as pl
from ber.pipeline import SEED, labels
from ber.stage2 import _chosen
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(260); pl.Config.set_fmt_str_lengths(40)
OUT = Path("/work/dive/orph"); W = Path("/work/run24"); TAG = "run24_both"
cfg = json.load(open(W / f"val_report_{TAG}.json"))["best"]
print("cfg", cfg, flush=True)
cols = ["entity_id", "src", "country", "name_n", "house", "addr_empty", "addr_w"]

lab = labels(Path("/work/data"), Path("/work/base")).with_columns(
    (pl.col("parent").fill_null(pl.col("rid")).hash(SEED + 7) % 2 == 0).alias("half_a"),
    pl.col("rid").cast(pl.UInt32), pl.col("parent").cast(pl.UInt32))
tr = pl.read_parquet("/work/base/train/records.parquet", columns=cols).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
tr = tr.with_columns(((pl.col("src") == 1) & ((pl.col("entity_id").hash(SEED + 77) % 100) < 19)).alias("hidden"))
gt = pl.read_parquet("/work/data/train_pairs.parquet")
e2r = tr.select("entity_id", "rid")
rawp = gt.join(e2r.rename({"entity_id": "m", "rid": "q"}), on="m").join(e2r.rename({"entity_id": "s1", "rid": "rawp"}), on="s1").select("q", "rawp")
rawp.write_parquet(OUT / "rawp.parquet")
# decoy words (train raw): >=90% of S2/S3 records carrying the word are raw orphans, >=2000 records
qq = tr.filter(pl.col("src") != 1).select("rid", "country", "name_n").join(rawp.rename({"q": "rid"}), on="rid", how="left")
w = qq.select("rid", "country", pl.col("rawp").is_null().alias("R"), pl.col("name_n").str.split(" ").alias("w")).explode("w").unique(["rid", "w"])
dec = w.group_by("country", "w").agg(pl.len().alias("n"), pl.col("R").mean().alias("rR")).filter((pl.col("n") >= 2000) & (pl.col("rR") >= 0.9))
decw = pl.concat([dec.select("country", "w"),
                  pl.DataFrame({"country": ["France"] * 4, "w": ["participations", "holding", "distribution", "international"]})])
decw.write_parquet(OUT / "decoy_words.parquet")
print("decoy words:", sorted(decw.rows()), flush=True)
del w, qq
# S1 name frequencies: pool (visible) and full list
s1 = tr.filter(pl.col("src") == 1)
n_pool = s1.filter(~pl.col("hidden")).group_by("country", "name_n").agg(pl.len().alias("n_pool"))
n_full = s1.group_by("country", "name_n").agg(pl.len().alias("n_full"))


def feats(best, rec, n_pool, n_full):
    """best: q, s1, prob -> + record/S1 relation features."""
    r = rec.select("rid", "country", "name_n", "house", "addr_empty", "addr_w", "src")
    d = best.join(r.rename({c: f"q_{c}" for c in r.columns if c != "rid"}).rename({"rid": "q"}), on="q") \
            .join(r.select("rid", "name_n", "house", "addr_w").rename({"rid": "s1", "name_n": "s_name_n", "house": "s_house", "addr_w": "s_addr_w"}), on="s1")
    d = d.join(n_pool.rename({"country": "q_country", "name_n": "s_name_n"}), on=["q_country", "s_name_n"], how="left") \
         .join(n_full.rename({"country": "q_country", "name_n": "s_name_n"}), on=["q_country", "s_name_n"], how="left")
    qt, st = pl.col("q_name_n").str.split(" "), pl.col("s_name_n").str.split(" ")
    d = d.with_columns(qt.list.set_difference(st).alias("q_extra"), st.list.set_difference(qt).alias("s_extra"))
    ne, se = pl.col("q_extra").list.len(), pl.col("s_extra").list.len()
    qa, sa = pl.col("q_addr_w").str.split(" "), pl.col("s_addr_w").str.split(" ")
    d = d.with_columns(
        pl.when(pl.col("q_name_n") == pl.col("s_name_n")).then(pl.lit("exact"))
          .when((se == 0) & (ne > 0)).then(pl.lit("adds")).when((ne == 0) & (se > 0)).then(pl.lit("drops"))
          .when((ne == 1) & (se == 1)).then(pl.lit("repl1")).otherwise(pl.lit("other")).alias("nrel"),
        pl.when(pl.col("q_addr_empty")).then(pl.lit("empty")).when(pl.col("q_house") == "").then(pl.lit("q_noh"))
          .when(pl.col("s_house") == "").then(pl.lit("s_noh")).when(pl.col("q_house") == pl.col("s_house")).then(pl.lit("h_eq"))
          .otherwise(pl.lit("h_diff")).alias("hrel"),
        pl.when(pl.col("q_addr_empty")).then(pl.lit("empty"))
          .when(pl.col("q_addr_w") == pl.col("s_addr_w")).then(pl.lit("st_eq"))
          .when(qa.list.set_intersection(sa).list.len() >= 0.5 * pl.max_horizontal(qa.list.len(), sa.list.len()))
          .then(pl.lit("st_sim")).otherwise(pl.lit("st_diff")).alias("srel"),
        pl.col("n_pool").fill_null(0), pl.col("n_full").fill_null(0))
    ex = d.select("q", pl.col("q_country").alias("country"), pl.col("q_extra").alias("w")).explode("w") \
          .join(decw, on=["country", "w"], how="semi")
    d = d.with_columns(pl.col("q").is_in(ex["q"].unique().implode()).alias("decoy"))
    d = d.with_columns(
        pl.when(pl.col("n_pool") == 0).then(pl.lit("u0")).when(pl.col("n_pool") == 1).then(pl.lit("u1"))
          .when(pl.col("n_pool") <= 4).then(pl.lit("u2-4")).otherwise(pl.lit("u5+")).alias("ufreq"))
    return d.drop("q_extra", "s_extra", "q_addr_w", "s_addr_w")


vb = pl.read_parquet(W / f"val_best_{TAG}.parquet")
chv = _chosen(vb, cfg).with_columns(pl.lit(True).alias("acc"))
vb = vb.join(chv, on=["q", "s1"], how="left").with_columns(pl.col("acc").fill_null(False))
v = feats(vb, tr, n_pool, n_full)
v = v.join(lab.select(pl.col("rid").alias("q"), "parent", pl.col("half_a").alias("q_half_a")), on="q", how="left") \
     .join(rawp, on="q", how="left") \
     .join(lab.filter(pl.col("src") == 1).select(pl.col("rid").alias("s1"), pl.col("is_val").alias("s1_val"), pl.col("half_a").alias("s1_half_a")), on="s1", how="left")
v = v.with_columns((pl.col("parent") == pl.col("s1")).fill_null(False).alias("y"),
                   pl.when(pl.col("parent").is_not_null()).then(pl.lit("M")).when(pl.col("rawp").is_not_null()).then(pl.lit("H")).otherwise(pl.lit("R")).alias("cls"))
# name of the hidden parent (is the candidate a namesake of the hidden parent?)
v = v.join(tr.select(pl.col("rid").alias("rawp"), pl.col("name_n").alias("hp_name")), on="rawp", how="left")
v.write_parquet(OUT / "val_feats.parquet")
print("val records (best pair):", v.height, v.group_by("cls").len().sort("cls").rows(), flush=True)
vv = v.filter(pl.col("s1_val"))  # best S1 is a validation S1 (what the metric sees)
print("best S1 is val S1:", vv.height, vv.group_by("cls").len().sort("cls").rows(), flush=True)
del tr

te = pl.read_parquet("/work/base/test/records.parquet", columns=cols).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
t_pool = te.filter(pl.col("src") == 1).group_by("country", "name_n").agg(pl.len().alias("n_pool"))
tb = pl.read_parquet(W / f"test_best_{TAG}.parquet")
cht = _chosen(tb, cfg).with_columns(pl.lit(True).alias("acc"))
tb = tb.join(cht, on=["q", "s1"], how="left").with_columns(pl.col("acc").fill_null(False))
t = feats(tb, te, t_pool, t_pool.rename({"n_pool": "n_full"}))
t.write_parquet(OUT / "test_feats.parquet")
del te
print("test records (best pair):", t.height, "accepted", t["acc"].sum(), flush=True)


def prof(d, label):
    return d.select(pl.lit(label).alias("set"), pl.len().alias("n"),
                    pl.col("q_addr_empty").mean().round(3).alias("empty"),
                    *[(pl.col("nrel") == k).mean().round(3).alias(k) for k in ["exact", "adds", "drops", "repl1", "other"]],
                    *[(pl.col("hrel") == k).mean().round(3).alias(k) for k in ["h_eq", "h_diff", "q_noh", "s_noh"]],
                    (pl.col("srel") == "st_eq").mean().round(3).alias("st_eq"), (pl.col("srel") == "st_diff").mean().round(3).alias("st_diff"),
                    pl.col("decoy").mean().round(3).alias("decoy"), (pl.col("q_src") == 3).mean().round(3).alias("src3"),
                    (pl.col("n_pool") == 1).mean().round(3).alias("s_u1"), (pl.col("n_pool") >= 5).mean().round(3).alias("s_u5+"),
                    pl.col("prob").mean().round(3).alias("mprob"), pl.col("acc").mean().round(3).alias("acc"))


rows = [prof(vv.filter(pl.col("cls") == "M"), "val M (parent visible)"),
        prof(vv.filter(pl.col("cls") == "H"), "val H hidden-parent orph"),
        prof(vv.filter(pl.col("cls") == "R"), "val R raw orphan"),
        prof(vv.filter(~pl.col("acc")), "val rejected (all)"),
        prof(vv.filter(~pl.col("acc") & (pl.col("prob") >= 0.05)), "val rejected p>=0.05"),
        prof(vv.filter(pl.col("prob").is_between(0.05, 0.95)), "val uncertain .05-.95"),
        prof(t.filter(~pl.col("acc")), "test rejected (all)"),
        prof(t.filter(~pl.col("acc") & (pl.col("prob") >= 0.05)), "test rejected p>=0.05"),
        prof(t.filter(pl.col("prob").is_between(0.05, 0.95)), "test uncertain .05-.95")]
for c in ["US", "India", "France"]:
    rows.append(prof(t.filter((pl.col("q_country") == c) & pl.col("prob").is_between(0.05, 0.95)), f"test unc {c}"))
    if c != "France":
        rows.append(prof(vv.filter((pl.col("q_country") == c) & pl.col("prob").is_between(0.05, 0.95)), f"val unc {c}"))
print(pl.concat(rows))
print(vv.group_by("q_country", "cls").len().sort("q_country", "cls"))
print(t.group_by("q_country").agg(pl.len(), pl.col("acc").sum().alias("acc"), pl.col("prob").is_between(0.05, 0.95).sum().alias("unc")).sort("q_country"))

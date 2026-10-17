"""Compact re-run of the orph1 signature table printing every column (per-S1 densities; train raw / train hid19 / test),
plus the same signatures per class (M/H/R) in train raw, and decoy-word densities."""
import polars as pl
SEED = 42
cols = ["entity_id", "src", "country", "name_n", "house", "addr_empty"]
FLAGS = ["name_in_s1", "name_uniq_s1", "nh_in_s1", "orph_cluster_name", "orph_cluster_nh", "addr_empty",
         "empty_noname", "empty_uniq", "empty_shared", "nonempty_noname", "nonempty_nh_noS1"]


def sig(q, s1pool):
    sn = s1pool.group_by("country", "name_n").agg(pl.len().alias("n_s1_name"))
    snh = s1pool.filter(pl.col("house") != "").group_by("country", "name_n", "house").agg(pl.len().alias("n_s1_nh"))
    qn = q.group_by("country", "name_n").agg(pl.len().alias("n_q_name"))
    qnh = q.filter(pl.col("house") != "").group_by("country", "name_n", "house").agg(pl.len().alias("n_q_nh"))
    d = q.join(sn, on=["country", "name_n"], how="left").join(snh, on=["country", "name_n", "house"], how="left") \
         .join(qn, on=["country", "name_n"], how="left").join(qnh, on=["country", "name_n", "house"], how="left") \
         .with_columns(pl.col("n_s1_name").fill_null(0), pl.col("n_s1_nh").fill_null(0), pl.col("n_q_nh").fill_null(0))
    e = pl.col("addr_empty")
    return d.with_columns(
        (pl.col("n_s1_name") > 0).alias("name_in_s1"), (pl.col("n_s1_name") == 1).alias("name_uniq_s1"),
        (pl.col("n_s1_nh") > 0).alias("nh_in_s1"),
        ((pl.col("n_s1_name") == 0) & (pl.col("n_q_name") >= 2)).alias("orph_cluster_name"),
        ((pl.col("n_s1_nh") == 0) & (pl.col("n_q_nh") >= 2) & (pl.col("n_s1_name") == 0)).alias("orph_cluster_nh"),
        (e & (pl.col("n_s1_name") == 0)).alias("empty_noname"), (e & (pl.col("n_s1_name") == 1)).alias("empty_uniq"),
        (e & (pl.col("n_s1_name") >= 2)).alias("empty_shared"), (~e & (pl.col("n_s1_name") == 0)).alias("nonempty_noname"),
        (~e & (pl.col("house") != "") & (pl.col("n_s1_nh") == 0)).alias("nonempty_nh_noS1"))


def dens(d, s1pool, by):
    ns1 = s1pool.group_by("country").agg(pl.len().alias("n_s1"))
    o = d.group_by(by).agg(pl.len().alias("recs"), *[pl.col(f).sum() for f in FLAGS]).join(ns1, on="country")
    return o.select(*by, "n_s1", *[(pl.col(f) / pl.col("n_s1")).round(4).alias(f) for f in ["recs"] + FLAGS]).sort(by)


def show(df, title):
    print(title, flush=True)
    for r in df.to_dicts():
        print("  ", r, flush=True)


tr = pl.read_parquet("/work/base/train/records.parquet", columns=cols)
gt = pl.read_parquet("/work/data/train_pairs.parquet")
s1 = tr.filter(pl.col("src") == 1).with_columns(((pl.col("entity_id").hash(SEED + 77) % 100) < 19).alias("hidden"))
q = tr.filter(pl.col("src") != 1).join(gt.rename({"m": "entity_id", "s1": "p_eid"}), on="entity_id", how="left")
q = q.join(s1.select(pl.col("entity_id").alias("p_eid"), pl.col("hidden").alias("p_hidden")), on="p_eid", how="left")
q = q.with_columns(pl.when(pl.col("p_eid").is_null()).then(pl.lit("R")).when(pl.col("p_hidden")).then(pl.lit("H")).otherwise(pl.lit("M")).alias("cls"))
del tr
d = sig(q, s1)
show(dens(d, s1, ["country"]), "train_raw (per S1)")
show(dens(d, s1, ["country", "cls"]), "train_raw by class (per S1)")
vis = s1.filter(~pl.col("hidden"))
d = sig(q, vis)
show(dens(d, vis, ["country"]), "train_hid19 (per visible S1)")
show(dens(d, vis, ["country", "cls"]), "train_hid19 by class (per visible S1)")
del d, q
te = pl.read_parquet("/work/base/test/records.parquet", columns=cols)
ts1 = te.filter(pl.col("src") == 1)
tq = te.filter(pl.col("src") != 1)
del te
show(dens(sig(tq, ts1), ts1, ["country"]), "test (per S1)")

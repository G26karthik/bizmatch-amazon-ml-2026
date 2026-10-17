"""Copies per S1 per source from the raw GT (light; runs anywhere): distributions, joint S2 x S3, independence."""
import numpy as np
import polars as pl
pl.Config.set_tbl_rows(30); pl.Config.set_tbl_width_chars(200)
tp = pl.read_parquet("local_data/train_pairs.parquet")
s1 = pl.read_parquet("local_data/train_source1.parquet", columns=["entity_id", "country"])
c = tp.with_columns(pl.col("m").str.slice(0, 2).alias("p")).group_by("s1").agg((pl.col("p") == "S2").sum().alias("n2"), (pl.col("p") == "S3").sum().alias("n3"))
s = s1.join(c, left_on="entity_id", right_on="s1", how="left").fill_null(0)
for C in ("US", "India"):
    d = s.filter(pl.col("country") == C)
    print(f"\n#### {C}: S1 {d.height}; mean n2 {d['n2'].mean():.3f} n3 {d['n3'].mean():.3f}; var n2 {d['n2'].var():.3f} n3 {d['n3'].var():.3f}; corr {np.corrcoef(d['n2'].to_numpy(), d['n3'].to_numpy())[0,1]:.4f}")
    for col in ("n2", "n3"):
        vc = d.group_by(col).len().sort(col); print(col, dict(zip(vc[col].to_list(), vc["len"].to_list())))
    t = (d["n2"] + d["n3"]); vc = t.value_counts().sort(t.name); print("n2+n3", dict(zip(vc[t.name].to_list(), vc["count"].to_list())))
    ct = d.group_by("n2", "n3").len()
    p2 = d.group_by("n2").len().with_columns((pl.col("len") / d.height).alias("p2")).drop("len")
    p3 = d.group_by("n3").len().with_columns((pl.col("len") / d.height).alias("p3")).drop("len")
    ct = ct.join(p2, on="n2").join(p3, on="n3").with_columns((pl.col("len") / (pl.col("p2") * pl.col("p3") * d.height)).round(2).alias("o_e"))
    print(ct.pivot(on="n3", index="n2", values="len", sort_columns=True).sort("n2").fill_null(0))
    print("obs/exp:"); print(ct.pivot(on="n3", index="n2", values="o_e", sort_columns=True).sort("n2").fill_null(0))

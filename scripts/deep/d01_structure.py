"""Deep dive 1: generator structure in train truth. Empty-address rates, copies per S1 (S2/S3 joint), and whether
copies of one S1 are adjacent in file order or correlated with the S1's row/id.

uv run --no-project --with polars==1.44.2 python scripts/deep/d01_structure.py
"""
import numpy as np
import polars as pl

pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
S = {s: pl.read_parquet(f"local_data/train_source{s}.parquet").with_row_index("row") for s in (1, 2, 3)}
tp = pl.read_parquet("local_data/train_pairs.parquet")
print("pairs", tp.height, "S1 in pairs", tp["s1"].n_unique(), "S1 total", S[1].height)
for s in (2, 3):
    d = S[s]
    print(f"S{s}: rows {d.height}, empty addr {(d['business_address'].fill_null('').str.strip_chars() == '').mean():.4f}, "
          f"in truth {d['entity_id'].is_in(tp['m'].implode()).mean():.4f}")
    print(d.with_columns((pl.col("business_address").fill_null("").str.strip_chars() == "").alias("empty"),
                         pl.col("entity_id").is_in(tp["m"].implode()).alias("copy"))
          .group_by("country").agg(pl.len(), pl.col("empty").mean().round(4), pl.col("copy").mean().round(4),
                                   pl.col("copy").filter(pl.col("empty")).mean().round(4).alias("copy|empty")).sort("country"))
# copies per S1 joint distribution
c = tp.with_columns(pl.col("m").str.slice(0, 2).alias("src")).group_by("s1").agg((pl.col("src") == "S2").sum().alias("n2"), (pl.col("src") == "S3").sum().alias("n3"))
c = S[1].select(pl.col("entity_id").alias("s1"), "country").join(c, on="s1", how="left").fill_null(0)
print("S1s with no copies:", (c["n2"] + c["n3"] == 0).mean())
print("n2 dist:", c.group_by("n2").len().sort("n2").rows())
print("n3 dist:", c.group_by("n3").len().sort("n3").rows())
j = c.group_by("n2", "n3").len().sort("n2", "n3")
print(j.pivot(on="n3", index="n2", values="len").fill_null(0))
print("corr n2,n3:", np.corrcoef(c["n2"].to_numpy(), c["n3"].to_numpy())[0, 1])
# adjacency: are consecutive rows in S2 copies of the same S1?
for s in (2, 3):
    m = S[s].select("row", pl.col("entity_id").alias("m")).join(tp, on="m", how="left").sort("row")
    s1 = m["s1"].to_list()
    same = sum(1 for a, b in zip(s1, s1[1:]) if a is not None and a == b)
    print(f"S{s}: consecutive rows same S1: {same} of {len(s1) - 1}")
    r1 = S[1].select(pl.col("entity_id").alias("s1"), pl.col("row").alias("r1"))
    mm = m.join(r1, on="s1").filter(pl.col("s1").is_not_null())
    print(f"S{s}: corr(row, S1 row) = {np.corrcoef(mm['row'].to_numpy(), mm['r1'].to_numpy())[0, 1]:.5f}")
    ida = mm["m"].str.slice(3).cast(pl.Int64).to_numpy(); idb = mm["s1"].str.slice(3).cast(pl.Int64).to_numpy()
    print(f"S{s}: corr(id, S1 id) = {np.corrcoef(ida, idb)[0, 1]:.5f}; id mod 1000 equal {np.mean(ida % 1000 == idb % 1000):.5f}")

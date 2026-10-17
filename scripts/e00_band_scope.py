"""E00 follow-up 2 (scope only, light): size of the Stage-A band 0.001 <= pA < 0.01 on train/test, emb_cos coverage,
and how the cross-encoder zone relates to pA."""
from pathlib import Path

import polars as pl

for split in ("train", "test"):
    d = Path(f"/work/run28/{split}/pruned")
    p = pl.scan_parquet(str(d / "*.parquet")).select("q", "s1", "pA", "country")
    s = p.group_by("country").agg(pl.len().alias("all_ge_0.001"), (pl.col("pA") >= 0.01).sum().alias("ge_0.01"),
                                  ((pl.col("pA") < 0.01)).sum().alias("band")).collect().sort("country")
    print(split, s)
    e = pl.scan_parquet(f"/work/run28/{split}/emb_cos_pruned.parquet").select(pl.len()).collect().item()
    print(f"{split}: emb_cos_pruned rows {e}")
    band = p.filter(pl.col("pA") < 0.01).select("q", "s1").head(200000).collect()
    ec = pl.read_parquet(f"/work/run28/{split}/emb_cos_pruned.parquet").select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
    cov = band.with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32)).join(ec, on=["q", "s1"], how="semi").height
    print(f"{split}: emb_cos coverage of a 200k sample of band pairs: {cov / band.height:.3f}")
    nq_band_only = p.group_by("q").agg(pl.col("pA").max().alias("m")).filter(pl.col("m") < 0.01).select(pl.len()).collect().item()
    print(f"{split}: records whose only candidates are band pairs (currently no candidate at all): {nq_band_only}")
ce = pl.read_parquet("/work/ce_wide/ce_scores_val.parquet").select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
pr = pl.scan_parquet("/work/run28/train/pruned/*.parquet").select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32), "pA").collect()
z = pr.join(ce, on=["q", "s1"], how="inner")
print(f"ce_wide val zone pairs {ce.height}: pA min {z['pA'].min():.4f}, share pA<0.01 {(z['pA'] < 0.01).mean():.4f}")

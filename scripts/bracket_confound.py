"""Is the bracket marker independent of the model probability? In pure classes (generic|same = copies, type|offset =
decoys) compare P(prob > t | bracket) vs P(prob > t | no bracket). Then a text-free cross-check of the prob split on
France same-house type swaps: cap test by prob bin (rate per 1k S1-source slots with 2-3 other fr7-accepted records vs at
the cap 5/6; copies vanish at the cap, decoys do not), plus finer bracket bins above 0.99.

python scripts/bracket_confound.py
"""
import polars as pl

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(230)
BC, BD = 0.0418, 0.0090
x = pl.read_parquet("/work/lead/fr_bracket_mix.parquet")
tb = pl.read_parquet("/work/run28/test_best_run28_7q.parquet").select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32), "prob")
x = x.join(tb, on=["q", "s1"], how="left")
print("pairs with a blended prob (record's best S1):", x.filter(pl.col("prob").is_not_null()).height, "of", x.height)
for cls in (("generic", "same"), ("type", "offset"), ("near", "same"), ("type", "same")):
    y = x.filter((pl.col("kind") == cls[0]) & (pl.col("hrel") == cls[1]) & pl.col("prob").is_not_null())
    t = y.group_by("bracket").agg(pl.len().alias("n"), (pl.col("prob") > 0.99).mean().round(4).alias("p>0.99"),
                                  (pl.col("prob") > 0.5).mean().round(4).alias("p>0.5"), pl.col("prob").mean().round(4).alias("mean_p")).sort("bracket")
    print(f"\n{cls}:"); print(t)
# finer bins, contested type|same
y = x.filter((pl.col("kind") == "type") & (pl.col("hrel") == "same") & pl.col("prob").is_not_null())
y = y.with_columns(pl.col("prob").cut([0.5, 0.9, 0.99, 0.995, 0.999, 0.9999]).alias("pbin"))
print("\ntype|same (all statuses) by prob bin, bracket decoy estimate")
print(y.group_by("pbin", "status").agg(pl.len().alias("n"), pl.col("bracket").sum().alias("n_brk"), pl.col("bracket").mean().round(4).alias("brk"))
       .with_columns(((BC - pl.col("brk")) / (BC - BD)).round(2).alias("d_brk")).sort("pbin", "status"))
# cap test by prob bin (fr7 acceptance for the n_other count; the type|same record itself excluded)
eid = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
g = pl.read_csv("/work/out_fr7/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = g.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "") \
       .join(eid.select(pl.col("entity_id").alias("source1_entity_id"), pl.col("rid").alias("s1")), on="source1_entity_id") \
       .join(eid.select(pl.col("entity_id").alias("matched_entity_ids"), pl.col("rid").alias("q"), "src"), on="matched_entity_ids").select("q", "s1", "src")
ts = x.filter((pl.col("kind") == "type") & (pl.col("hrel") == "same")).select("q").unique()
oth = acc.join(ts, on="q", how="anti").group_by("s1", "src").agg(pl.len().alias("n_other"))
y = y.with_columns(pl.col("src").cast(pl.Int8)).join(oth.with_columns(pl.col("src").cast(pl.Int8)), on=["s1", "src"], how="left").with_columns(pl.col("n_other").fill_null(0))
s1f = eid.filter((pl.col("src") == 1) & (pl.col("country") == "France")).select(pl.col("rid").alias("s1"))
den = pl.concat([s1f.with_columns(pl.lit(s, dtype=pl.Int8).alias("src")) for s in (2, 3)]) \
        .join(oth.with_columns(pl.col("src").cast(pl.Int8)), on=["s1", "src"], how="left").with_columns(pl.col("n_other").fill_null(0))
capc = ((pl.col("src") == 2) & (pl.col("n_other") >= 5)) | ((pl.col("src") == 3) & (pl.col("n_other") >= 6))
d_cap = den.filter(capc).height; d_23 = den.filter(pl.col("n_other").is_between(2, 3)).height
y = y.with_columns(pl.col("prob").cut([0.5, 0.99, 0.999]).alias("pb4"))
t = y.group_by("pb4").agg(pl.len().alias("n"), pl.col("n_other").is_between(2, 3).sum().alias("n23"), capc.sum().alias("ncap")) \
     .with_columns((pl.col("n23") / d_23 * 1000).round(3).alias("r23"), (pl.col("ncap") / d_cap * 1000).round(3).alias("rcap")) \
     .with_columns((pl.col("rcap") / pl.col("r23")).round(3).alias("cap_ratio")).sort("pb4")
print(f"\ncap test, type|same by prob (slots: n23 {d_23}, cap {d_cap}); copies -> ratio ~0, decoys -> ratio ~ decoy reference below")
print(t)
z = x.filter((pl.col("kind") == "type") & (pl.col("hrel") == "offset")).with_columns(pl.col("src").cast(pl.Int8)) \
     .join(oth.with_columns(pl.col("src").cast(pl.Int8)), on=["s1", "src"], how="left").with_columns(pl.col("n_other").fill_null(0))
zz = z.select(pl.col("n_other").is_between(2, 3).sum().alias("n23"), capc.sum().alias("ncap"))
print("decoy reference type|offset cap ratio:", round((zz["ncap"][0] / d_cap) / (zz["n23"][0] / d_23), 3))
zc = x.filter((pl.col("kind") == "generic") & (pl.col("hrel") == "same")).with_columns(pl.col("src").cast(pl.Int8)) \
      .join(oth.with_columns(pl.col("src").cast(pl.Int8)), on=["s1", "src"], how="left").with_columns(pl.col("n_other").fill_null(0))
zz = zc.select(pl.col("n_other").is_between(2, 3).sum().alias("n23"), capc.sum().alias("ncap"))
print("copy reference generic|same cap ratio:", round((zz["ncap"][0] / d_cap) / (zz["n23"][0] / d_23), 3))

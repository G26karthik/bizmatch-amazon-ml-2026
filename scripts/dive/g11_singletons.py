"""Generator dive 11 (worker): unmatched records whose best full-pool S1 has NO true copy (singleton S1s) vs others."""
import polars as pl

pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(230); pl.Config.set_fmt_str_lengths(60)
s = pl.read_parquet("/work/dive/s1_counts.parquet").select(pl.col("s1r").cast(pl.UInt32), (pl.col("n2") + pl.col("n3")).alias("ncop"), "nd_any")
for C in ("US", "India"):
    a = pl.read_parquet(f"/work/dive/allq_{C}.parquet", columns=["mr", "s1r", "src", "pA", "matched", "y", "name_n_1", "name_n_2", "house_1",
                                                                "house_2", "addr_empty_2", "addr_w_1", "addr_w_2", "business_name_1", "business_name_2"])
    a = a.join(s, on="s1r", how="left").filter(~pl.col("matched"))
    a = a.with_columns((pl.col("ncop") == 0).alias("single"),
                       (pl.col("house_1").cast(pl.Int64, strict=False) - pl.col("house_2").cast(pl.Int64, strict=False)).abs().alias("hd"))
    print(f"\n#### {C} unmatched records by whether best S1 is a singleton (no true copies)")
    print(a.group_by("single").agg(pl.len(), pl.col("pA").mean().round(3), (pl.col("pA") >= 0.5).mean().round(4).alias("pA>=.5"),
          (pl.col("name_n_1") == pl.col("name_n_2")).mean().round(4).alias("name_eq"),
          ((pl.col("house_1") == pl.col("house_2")) & (pl.col("house_1") != "")).mean().round(4).alias("house_eq"),
          pl.col("hd").is_between(1, 50).mean().round(4).alias("hd_1_50"),
          (pl.col("addr_w_1") == pl.col("addr_w_2")).mean().round(4).alias("same_words"),
          pl.col("addr_empty_2").mean().round(4).alias("q_empty"),
          pl.col("nd_any").mean().round(2)).sort("single"))
    z = a.filter(pl.col("single") & (pl.col("pA") >= 0.3)).sample(min(25, a.height), seed=1)
    for r in z.iter_rows(named=True):
        print(f"S{r['src']} pA={r['pA']:.2f} nd={r['nd_any']} | {r['business_name_1']!r:36} -> {r['business_name_2']!r:40} | h {r['house_1']!r}->{r['house_2']!r} | {r['addr_w_1'][:35]!r} -> {(r['addr_w_2'] or '')[:35]!r}")

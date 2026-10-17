"""Raw samples of decoys (raw-unmatched) next to their best full-pool S1 candidate, by pA band."""
import polars as pl

for C in ("US", "India"):
    a = pl.read_parquet(f"/work/dive/allq_{C}.parquet", columns=["mr", "s1r", "src", "pA", "matched", "business_name_1", "business_name_2",
                                                                "addr_w_1", "addr_w_2", "house_1", "house_2", "addr_empty_2"])
    d = a.filter(~pl.col("matched"))
    for lo, hi in ((0.5, 1.01), (0.05, 0.5), (0.0, 0.05)):
        s = d.filter((pl.col("pA") >= lo) & (pl.col("pA") < hi))
        print(f"\n##### {C} decoys with best pA in [{lo},{hi}): {s.height} ({s.height / max(d.height, 1):.3f} of decoys)")
        for r in s.sample(min(14, s.height), seed=3).iter_rows(named=True):
            print(f"S{r['src']} pA={r['pA']:.2f} | {r['business_name_1']!r:38} -> {r['business_name_2']!r:44} | "
                  f"h {r['house_1']!r}->{r['house_2']!r} | {r['addr_w_1'][:40]!r} -> {(r['addr_w_2'] or '')[:40]!r}")

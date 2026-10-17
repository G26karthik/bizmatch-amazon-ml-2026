"""fr13 France edit lists from the bracket / cap evidence (bracket_mix.py, bracket_split.py, bracket_confound.py).
Contested = same-house type-word swaps (exact address: bracket_mix type|same; street typo: fr_swap_subsets C_edit_type).
  restore: contested pairs accepted in fr7, removed in fr12, blended prob > P_HI  (bracket share 5.5%, cap ratio 0.025 -> copies)
  remove : contested pairs kept in fr12 as the S1's only match, prob <= P_HI     (bracket share ~0.5% -> decoys)
Prints the cap test and bracket share for each list and for (0.999, P_HI] as a control.

python scripts/fr13_lists.py [P_HI=0.9999]
"""
import sys

import polars as pl

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(230)
P_HI = float(sys.argv[1]) if len(sys.argv) > 1 else 0.9999
rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country", "business_name", "addr_n"]).with_row_index("rid") \
        .with_columns(pl.col("rid").cast(pl.UInt32))
eid = rec.select("rid", "entity_id")


def accepted(out):
    g = pl.read_csv(f"{out}/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
    return g.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "") \
            .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
            .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1")


a7, a12 = accepted("/work/out_fr7"), accepted("/work/out_fr12")
bm = pl.read_parquet("/work/lead/fr_bracket_mix.parquet").filter((pl.col("kind") == "type") & (pl.col("hrel") == "same")).select("q", "s1", "bracket")
cb = pl.read_parquet("/work/lead/fr_swap_brk.parquet").filter(pl.col("cat") == "C_edit_type").select("q", "s1", "bracket")
cont = pl.concat([bm, cb]).unique(["q", "s1"])
tb = pl.read_parquet("/work/run28/test_best_run28_7q.parquet").select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32), "prob")
cont = cont.join(tb, on=["q", "s1"], how="left").join(rec.select(pl.col("rid").alias("q"), "src"), on="q")
cont = cont.join(a7.with_columns(pl.lit(True).alias("k7")), on=["q", "s1"], how="left").join(a12.with_columns(pl.lit(True).alias("k12")), on=["q", "s1"], how="left") \
           .with_columns(pl.col("k7").fill_null(False), pl.col("k12").fill_null(False))
n12 = a12.group_by("s1").len().rename({"len": "n12"})
cont = cont.join(n12, on="s1", how="left").with_columns(pl.col("n12").fill_null(0))
restore = cont.filter(pl.col("k7") & ~pl.col("k12") & (pl.col("prob") > P_HI))
restore = restore.join(a12.select("q"), on="q", how="anti")                      # record not assigned elsewhere in fr12
control = cont.filter(pl.col("k7") & ~pl.col("k12") & pl.col("prob").is_between(0.999, P_HI, closed="right"))
remove = cont.filter(pl.col("k12") & (pl.col("n12") == 1) & (pl.col("prob").fill_null(0) <= P_HI))
# cap test: other fr7-accepted records of the S1 in the same source, contested records excluded
oth = a7.join(cont.select("q"), on="q", how="anti").join(rec.select(pl.col("rid").alias("q"), "src"), on="q").group_by("s1", "src").agg(pl.len().alias("n_other"))
s1f = rec.filter((pl.col("src") == 1) & (pl.col("country") == "France")).select(pl.col("rid").alias("s1"))
den = pl.concat([s1f.with_columns(pl.lit(s, dtype=pl.Int8).alias("src")) for s in (2, 3)]).join(oth, on=["s1", "src"], how="left").with_columns(pl.col("n_other").fill_null(0))
capc = ((pl.col("src") == 2) & (pl.col("n_other") >= 5)) | ((pl.col("src") == 3) & (pl.col("n_other") >= 6))
d_cap, d_23 = den.filter(capc).height, den.filter(pl.col("n_other").is_between(2, 3)).height
for name, df in (("restore", restore), ("control (0.999, P_HI]", control), ("remove", remove)):
    z = df.join(oth, on=["s1", "src"], how="left").with_columns(pl.col("n_other").fill_null(0))
    n23, ncap = z.filter(pl.col("n_other").is_between(2, 3)).height, z.filter(capc).height
    ratio = (ncap / d_cap) / (n23 / d_23) if n23 else float("nan")
    print(f"{name:22} n {df.height:6}  bracket {df['bracket'].mean():.4f} ({df['bracket'].sum()})  cap n23 {n23} cap {ncap} ratio {ratio:.3f}")
names = dict(zip(rec["rid"].to_list(), rec["business_name"].to_list())); adr = dict(zip(rec["rid"].to_list(), rec["addr_n"].to_list()))
for name, df in (("restore", restore), ("remove", remove)):
    print(f"\nexamples {name}")
    for q, s, p in df.select("q", "s1", "prob").sample(n=min(10, df.height), seed=1).iter_rows():
        print(f"   {p:.6f}  {names[s]!r:42} | {names[q]!r:42} | {adr[s]}")
restore.select("q", "s1", "prob").write_parquet("/work/lead/fr13_restore.parquet")
remove.select("q", "s1", "prob").write_parquet("/work/lead/fr13_remove.parquet")
print("\nwrote /work/lead/fr13_restore.parquet", restore.height, "/work/lead/fr13_remove.parquet", remove.height)

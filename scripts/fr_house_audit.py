"""Accepted France pairs whose house number differs from the S1's: by name pattern (fr_name_ops_audit.parquet) and house
relation (decoy offset +{1,2,3,4,5,7,9,11,13,21}, other positive, negative). French copies almost never change the house
number; decoys sit at offsets. Bracket share where a new word exists; examples.

python scripts/fr_house_audit.py [--unaccepted]   (unaccepted: best-S1 pairs with prob > 0.1 not accepted, all address relations)
"""
import sys
import polars as pl

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(230)
OFFS = [1, 2, 3, 4, 5, 7, 9, 11, 13, 21]
UNACC = "--unaccepted" in sys.argv
x = pl.read_parquet("/work/lead/fr_name_ops_audit" + ("_unacc" if UNACC else "") + ".parquet")
x = x if UNACC else x.filter(pl.col("arel") == "diff_house")
tb = pl.read_parquet("/work/run28/test_best_run28_7q.parquet").select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32), "prob")
x = x.join(tb, on=["q", "s1"], how="left")
rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["business_name", "addr_n", "house"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
hs = rec["house"].to_list(); nm = rec["business_name"].to_list(); ad = rec["addr_n"].to_list()
dh = []
for q, s in zip(x["q"].to_list(), x["s1"].to_list()):
    try:
        dh.append(int(hs[q]) - int(hs[s]))
    except (TypeError, ValueError):
        dh.append(None)
x = x.with_columns(pl.Series("dh", dh, dtype=pl.Int64))
x = x.with_columns(pl.when(pl.col("arel") != "diff_house").then(pl.col("arel")).when(pl.col("dh").is_in(OFFS)).then(pl.lit("offset")).when(pl.col("dh") > 0).then(pl.lit("pos_other"))
                   .when(pl.col("dh") < 0).then(pl.lit("negative")).otherwise(pl.lit("nonnum")).alias("hrel"))
print("France pairs:", x.height)
print(x.group_by("pat", "hrel").agg(pl.len().alias("n"), pl.col("single").mean().round(3).alias("single"), pl.col("bracket").mean().round(4).alias("brk"),
      pl.col("prob").mean().round(3).alias("mean_p"), (pl.col("prob") > 0.5).mean().round(3).alias("p>0.5"))
       .filter(pl.col("n") >= 20).sort("n", descending=True))
print(x.group_by("dh").len().sort("len", descending=True).head(25).rows())
for h in (("negative", "pos_other", "no_addr", "no_house") if UNACC else ("offset", "negative", "pos_other")):
    print(f"\nexamples identical_key {h}")
    for q, s in x.filter((pl.col("pat") == "identical_key") & (pl.col("hrel") == h)).sample(n=10, seed=5, shuffle=True).select("q", "s1").iter_rows():
        print(f"   {nm[s]!r:40} | {nm[q]!r:40} | {ad[s]} | {ad[q]}")
x.write_parquet("/work/lead/fr_house_audit" + ("_unacc" if UNACC else "") + ".parquet")

"""Source-cap test for same-address one-word swaps. The generator caps copies per S1 at 5 in S2 and 6 in S3. For S1s whose
OTHER records in a source (accepted / true, excluding the swap records) already reach the cap, a swap record in that
source can only be a decoy. Compare the rate of swap records per S1 by the S1's other-record count in that source:
copies vanish at the cap, decoys do not. US train truth validates (true vs false swaps); France test uses accepted
non-swap records as the copy count.

python scripts/cap_test.py /work/out_fr8b
"""
import sys

import polars as pl

sys.argv = sys.argv[:1] + [sys.argv[1] if len(sys.argv) > 1 else "/work/out_fr8b"]
out = sys.argv[1]
src = open("/work/code/scripts/swap_verdict.py").read().split("def verdict")[0]
ns = {"__name__": "x"}
exec(src, ns)
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(200)
GEN_FR = ["fils", "cie", "associes", "services", "developpement", "groupe", "france", "frs", "freres", "compagnie"]
GEN_US = ["center", "services", "service", "partners", "centre"]


def table(sw, other, rec, label):
    """sw: swap records (q, s1, cls); other: (q, s1) copy records (excluding swaps). Count per S1 & source."""
    srcs = rec.select(pl.col("rid").alias("q"), "src")
    oc = other.join(srcs, on="q").group_by("s1", "src").agg(pl.len().alias("n_other"))
    x = sw.join(srcs, on="q").join(oc, on=["s1", "src"], how="left").with_columns(pl.col("n_other").fill_null(0))
    # base: number of S1s at each n_other level per source (denominator)
    s1s = rec.filter(pl.col("src") == 1).select(pl.col("rid").alias("s1"))
    base = pl.concat([s1s.with_columns(pl.lit(s, dtype=pl.Int8).alias("src")) for s in (2, 3)]).join(oc, on=["s1", "src"], how="left") \
             .with_columns(pl.col("n_other").fill_null(0)).group_by("src", "n_other").agg(pl.len().alias("n_s1"))
    t = x.group_by("src", "n_other", "cls").agg(pl.len().alias("swaps")).join(base, on=["src", "n_other"]) \
         .with_columns((pl.col("swaps") / pl.col("n_s1") * 1000).round(2).alias("per1k_S1"))
    print(f"\n=== {label}: swap records per 1k S1 by the S1's other-record count in the same source (cap S2=5, S3=6)")
    print(t.pivot(on="cls", index=["src", "n_other"], values="per1k_S1").sort("src", "n_other"))


# US train truth
tr = ns["prep"]("/work/base/train/records.parquet").filter(pl.col("country") == "US")
d = ns["swaps"](tr).filter(pl.col("arel") == "same")
gt = pl.read_parquet("/work/data/train_pairs.parquet")
e = tr.select("rid", "entity_id")
tp = gt.join(e.rename({"entity_id": "s1", "rid": "s1r"}), on="s1").join(e.rename({"entity_id": "m", "rid": "q"}), on="m").select("q", pl.col("s1r").alias("s1"))
d = d.join(tp.with_columns(pl.lit(True).alias("true")), on=["q", "s1"], how="left").with_columns(pl.col("true").fill_null(False))
d = d.with_columns(pl.when(pl.col("true")).then(pl.lit("TRUE")).otherwise(pl.lit("FALSE")).alias("cls"))
other = tp.join(d.select("q").unique(), on="q", how="anti")
table(d.select("q", "s1", "cls"), other, tr, "US train truth (other = true copies)")
# France test
te = ns["prep"]("/work/runfr/test/records.parquet").filter(pl.col("country") == "France")
f = pl.read_parquet("/work/dive/fr_swaps.parquet").filter(pl.col("arel") == "same") \
      .with_columns(pl.when(pl.col("w_in").is_in(GEN_FR)).then(pl.lit("GEN")).otherwise(pl.lit("TYPE")).alias("cls"))
rec_ids = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
g = pl.read_csv(f"{out}/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls().filter(pl.col("matched_entity_ids") != "") \
       .join(rec_ids.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
       .join(rec_ids.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1")
other_fr = acc.join(f.select("q").unique(), on="q", how="anti")
table(f.select("q", "s1", "cls"), other_fr, te, "France test (other = accepted non-swap records)")

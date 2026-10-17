"""Consistency check for removing France same-address type-word swaps (sibling decoys): per-S1 accepted-count
distribution before/after removal vs the generator's copy distribution (train truth, identical for US and India).
Also counts removals per S1 and how many S1s would become empty.

python scripts/sibling_remove_check.py /work/out_fr7
"""
import sys

import polars as pl

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(200)
out = sys.argv[1]
GEN = ["fils", "cie", "associes", "services", "developpement", "groupe", "france", "frs", "freres", "compagnie"]
sw = pl.read_parquet("/work/dive/fr_swaps.parquet")  # q, s1, arel, w_in (France one-swap records on the S1 street)
rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
eid = rec.select("rid", "entity_id")
g = pl.read_csv(f"{out}/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls("matched_entity_ids").filter(pl.col("matched_entity_ids") != "") \
       .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
       .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1")
fr_s1 = rec.filter((pl.col("country") == "France") & (pl.col("src") == 1)).select(pl.col("rid").alias("s1"))
acc_fr = acc.join(fr_s1, on="s1")
rm = sw.filter((pl.col("arel") == "same") & ~pl.col("w_in").is_in(GEN)).select("q", "s1").unique().join(acc_fr, on=["q", "s1"])
print("France accepted pairs:", acc_fr.height, "| same-address type-word swap pairs accepted (to remove):", rm.height)
print("removals per S1:", rm.group_by("s1").agg(pl.len().alias("n")).group_by("n").agg(pl.len().alias("s1s")).sort("n").rows())
after = acc_fr.join(rm, on=["q", "s1"], how="anti")
truth = {0: 0.0558, 1: 0.0540, 2: 0.1700, 3: 0.2405, 4: 0.2194}


def dist(p, label):
    k = fr_s1.join(p.group_by("s1").len(), on="s1", how="left").with_columns(pl.col("len").fill_null(0))
    d = k.group_by(pl.col("len").alias("k")).agg(pl.len().alias("n")).with_columns((pl.col("n") / k.height).round(4).alias(label)).sort("k")
    return d.select("k", label), k["len"].mean()


a, ma = dist(acc_fr, "before")
b, mb = dist(after, "after")
t = a.join(b, on="k", how="full", coalesce=True).with_columns(pl.col("k").replace_strict(truth, default=None).alias("truth")).sort("k")
print(t)
print(f"mean copies per S1: before {ma:.3f}, after {mb:.3f}, truth 3.46")
rm.write_parquet("/work/dive/fr_sibling_remove.parquet")

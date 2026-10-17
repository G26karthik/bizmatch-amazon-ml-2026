"""fr12 step of the France chain (was run inline on 27 Sep): from the fr10 submission, remove same-house type-word swaps
with a street typo (cap_test2.py category C_edit_type, accepted in fr10) where the S1 keeps another match.
Needs /work/dive/fr_swap_subsets.parquet from `python scripts/cap_test2.py <fr10 out>`.

python scripts/france_fr12.py /work/out_fr10 /work/out_fr12
"""
import shutil
import sys
from pathlib import Path

import polars as pl

base, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(exist_ok=True)
rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country"]).with_row_index("rid") \
        .with_columns(pl.col("rid").cast(pl.UInt32))
eid = rec.select("rid", "entity_id")
g = pl.read_csv(base / "matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls().filter(pl.col("matched_entity_ids") != "") \
       .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
       .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1")
sub = pl.read_parquet("/work/dive/fr_swap_subsets.parquet")
rm = sub.filter((pl.col("cat") == "C_edit_type") & pl.col("acc")).select("q", "s1").unique().join(acc, on=["q", "s1"])
k = acc.group_by("s1").agg(pl.len().alias("k"))
r = rm.group_by("s1").agg(pl.len().alias("r"))
rm = rm.join(k, on="s1").join(r, on="s1").filter(pl.col("k") > pl.col("r"))
print("remove C_edit_type (S1 keeps another match):", rm.height)
new = acc.join(rm.select("q", "s1"), on=["q", "s1"], how="anti")
s1all = rec.filter(pl.col("src") == 1).select(pl.col("rid").alias("s1"), pl.col("entity_id").alias("source1_entity_id"))
m = new.join(eid.rename({"rid": "q", "entity_id": "m"}), on="q").group_by("s1").agg(pl.col("m").sort().str.join(",").alias("matched_entity_ids"))
o = s1all.join(m, on="s1", how="left").with_columns(pl.col("matched_entity_ids").fill_null("")).select("source1_entity_id", "matched_entity_ids")
o = o.join(g.select("source1_entity_id").with_row_index("ord"), on="source1_entity_id").sort("ord").drop("ord")
with open(out / "matching_results.tsv", "w", encoding="utf-8", newline="\n") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for s, mm in o.iter_rows():
        f.write(f"{s}\t{mm}\n")
shutil.copy(base / "candidate_pairs.tsv", out / "candidate_pairs.tsv")
fr = rec.filter((pl.col("country") == "France") & (pl.col("src") == 1)).select(pl.col("rid").alias("s1"))
kk = fr.join(new.group_by("s1").agg(pl.len().alias("k")), on="s1", how="left").with_columns(pl.col("k").fill_null(0))
print("France pairs", new.join(fr, on="s1").height, "empty", (kk["k"] == 0).sum())

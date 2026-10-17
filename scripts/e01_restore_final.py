"""E01: finalise the non-word restore list (fr10/fr12 removals whose swapped-in word is not a word) with the swap in the
reason, a same-first-letter tier flag and a cap check against fr12 + restores (S2 <= 5, S3 <= 6 per S1).

python scripts/e01_restore_final.py
"""
from pathlib import Path

import polars as pl

OUT = Path("/work/agent_e01")
r = pl.read_parquet(OUT / "fr_nonword_restore.parquet").select("q", "s1", "prob")
w = pl.read_parquet(OUT / "fr12_removed_nonword.parquet").select("q", "s1", "w_in", "w_out", "same_first")
r = r.join(w, on=["q", "s1"], how="left")
rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country"]).with_row_index("rid").filter(pl.col("country") == "France")
eid = dict(zip(rec["entity_id"].to_list(), rec["rid"].to_list()))
src = rec.select(pl.col("rid").cast(pl.UInt32).alias("q"), "src")
rows = []
for line in open("/work/out_fr12/matching_results.tsv"):
    a, _, b = line.rstrip("\n").partition("\t")
    if a in eid and b:
        rows.extend((eid[x], eid[a]) for x in b.split(",") if x in eid)
acc = pl.DataFrame(rows, schema={"q": pl.UInt32, "s1": pl.UInt32}, orient="row")
assert r.join(acc.select("q"), on="q", how="semi").height == 0, "some restore records are assigned in fr12"
allp = pl.concat([acc, r.select("q", "s1")]).join(src, on="q")
cnt = allp.group_by("s1", "src").len()
over = cnt.filter(((pl.col("src") == 2) & (pl.col("len") > 5)) | ((pl.col("src") == 3) & (pl.col("len") > 6)))
print(f"restore {r.height} (same first letter {r['same_first'].sum()}); S1-source slots over cap after restore: {over.height}")
print(r.group_by("same_first").agg(pl.len(), pl.col("prob").median(), (pl.col("prob") < 0.5).sum().alias("p<0.5")))
print(r.group_by("w_out").len().sort("len", descending=True).head(15).rows())
r = r.with_columns(pl.concat_str(pl.lit("nonword_swap:"), pl.col("w_out"), pl.lit(">"), pl.col("w_in"),
                                 pl.when(pl.col("same_first")).then(pl.lit("|same_first")).otherwise(pl.lit("|diff_first"))).alias("reason"))
r.select("q", "s1", pl.col("prob").cast(pl.Float32), "reason", "same_first").write_parquet(OUT / "fr_nonword_restore.parquet")
print("wrote", OUT / "fr_nonword_restore.parquet")

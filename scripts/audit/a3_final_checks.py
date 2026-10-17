"""fr43 integrity: records assigned twice, per-S1 cap excess, presence of tie-breaker picks, added-vs-fr22 counts."""
import polars as pl
pl.Config.set_ascii_tables(True); pl.Config.set_tbl_rows(-1)
def load(v):
    g = pl.read_csv(f"output/{v}/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
    return (g.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
            .filter(pl.col("matched_entity_ids") != "").select(s1="source1_entity_id", m="matched_entity_ids"))
a43, a22 = load("v28q_fr43"), load("v28q_fr22")
print("fr22 pairs", a22.height, "fr43 pairs", a43.height)
d = a43.group_by("m").len().filter(pl.col("len") > 1); print("records with >1 S1 in fr43:", d.height)
c = a43.with_columns(src=pl.col("m").str.slice(0, 2)).group_by("s1", "src").len()
print("S1 per-source count dist (max):", c.group_by("src").agg(pl.col("len").max(), (pl.col("len") > 5).sum().alias("gt5"), (pl.col("len") > 6).sum().alias("gt6")))
new = a43.join(a22, on=["s1", "m"], how="anti"); gone = a22.join(a43, on=["s1", "m"], how="anti")
print("new vs fr22", new.height, "removed vs fr22", gone.height)
r = pl.read_parquet("local_data/work/audit/pick_bands.parquet").unique(["m", "s1", "step"])
r = r.join(a43.with_columns(in43=pl.lit(True)), on=["m", "s1"], how="left")
print(r.group_by("step").agg(pl.len(), pl.col("in43").sum()).sort("step"))
# dup across steps
print("picks in >1 step:", r.group_by("m").agg(pl.col("step").n_unique()).filter(pl.col("step") > 1).height)

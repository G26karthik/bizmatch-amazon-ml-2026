"""agent_mo task C: write adds for passing classes (train precision >= 0.85, no decoy excess), US/India only,
unassigned-in-fr27 records, respecting caps (<= 5 S2 / <= 6 S3 per S1 incl. fr27's existing matches).
Usage: python c_write_adds.py 'A:US:op=drop+drop' ...   (kind spec: task:country:column=value[,column=value])
Precision estimate per class is taken from the train tables (passed as prec=...).

uv run --no-project --with polars==1.44.2 python scripts/agent_mo/c_write_adds.py <specs>
"""
import sys

import polars as pl

picks = []
for spec in sys.argv[1:]:
    task, ctry, conds = spec.split(":")
    kv = dict(c.split("=") for c in conds.split(","))
    prec = float(kv.pop("prec"))
    te = pl.read_parquet(f"local_data/work/agent_mo_{task}_test_{ctry}.parquet").filter((pl.col("status") == "unassigned") & (pl.col("country") == ctry))
    for k, v in kv.items():
        te = te.filter(pl.col(k).cast(pl.String) == v)
    picks.append(te.select("m", "s1", "country", pl.lit(f"{task}:{conds}").alias("kind"), pl.lit(prec).alias("prec")))
if not picks:
    print("nothing passes; no file written"); sys.exit(0)
p = pl.concat(picks).unique("m", keep="first")
g = pl.read_csv("output/v28q_fr27/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = g.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "m"})
assert p.join(acc, on="m", how="semi").height == 0
have = acc.join(p.select("s1").unique(), on="s1", how="semi").group_by("s1", pl.col("m").str.slice(0, 2).alias("src")).len().rename({"len": "have"})
p = p.with_columns(pl.col("m").str.slice(0, 2).alias("src")).join(have, on=["s1", "src"], how="left").with_columns(pl.col("have").fill_null(0))
p = p.with_columns((pl.col("have") + pl.int_range(1, pl.len() + 1).over("s1", "src")).alias("tot"))
cap = pl.when(pl.col("src") == "S2").then(5).otherwise(6)
print("dropped by caps:", p.filter(pl.col("tot") > cap).height)
p = p.filter(pl.col("tot") <= cap).select("m", "s1", "country", "kind", "prec")
p.write_parquet("local_data/work/agent_mo_adds.parquet")
print(p.group_by("country", "kind").agg(pl.len(), pl.col("prec").first()).sort("country", "kind"))

"""Full-cap test for French same-name same-street records by house relation (house_jump_truth.py output), unassigned
ones only (model-free): share of records whose S1 already holds a full source cap (5 S2 / 6 S3 accepted in the base
submission) vs the share of all S1-source slots that are full (what a population unrelated to the S1's copies shows).
Copies cannot sit at a full cap; decoys do so at the background rate. Reference classes: offsets (decoys).

uv run --no-project --with polars==1.44.2 python scripts/jump_fullcap.py
"""
from collections import Counter
from pathlib import Path

import polars as pl

SP = Path("local_data/work")
x = pl.read_parquet(SP / "fr_house_jump.parquet")            # m, s1, hrel, acc (fr14 acceptance)
g = pl.read_csv("output/v28q_fr16/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
cnt, assigned = Counter(), set()
fr_s1 = set(pl.read_parquet("local_data/test_source1.parquet").filter(pl.col("country") == "France")["entity_id"].to_list())
for s, m in zip(g["source1_entity_id"].to_list(), g["matched_entity_ids"].fill_null("").to_list()):
    for e in (m.split(",") if m else []):
        cnt[(s, e[:2])] += 1; assigned.add(e)
full = lambda s, src: cnt[(s, src)] >= (5 if src == "S2" else 6)  # noqa: E731
tot = {src: sum(1 for s in fr_s1 if full(s, src)) / len(fr_s1) for src in ("S2", "S3")}
print("background share of France S1s with a full cap:", {k: round(v, 4) for k, v in tot.items()})
x = x.with_columns(pl.Series("assigned", [m in assigned for m in x["m"].to_list()]),
                   pl.Series("src", [m[:2] for m in x["m"].to_list()]),
                   pl.Series("full", [full(s, m[:2]) for m, s in zip(x["m"].to_list(), x["s1"].to_list())]))
u = x.filter(~pl.col("assigned"))
t = u.group_by("hrel", "src").agg(pl.len().alias("n_unassigned"), pl.col("full").sum().alias("at_full_cap")).sort("hrel", "src")
t = t.with_columns(pl.Series("expected_if_unrelated", [round(n * tot[s], 1) for n, s in zip(t["n_unassigned"].to_list(), t["src"].to_list())]))
print(t)

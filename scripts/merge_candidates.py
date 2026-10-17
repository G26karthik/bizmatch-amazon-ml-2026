"""Make candidate_pairs.tsv a superset of matching_results.tsv (rule-added matches, e.g. initials / generic-swap copies,
must appear among the audited candidates). Rewrites <out>/candidate_pairs.tsv in place, keeping row order.

python scripts/merge_candidates.py /work/out_fr9
"""
import sys
from pathlib import Path

import polars as pl

out = Path(sys.argv[1])
m = pl.read_csv(out / "matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
c = pl.read_csv(out / "candidate_pairs.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "candidate_entity_ids": pl.String})
ex = lambda df, col: df.with_columns(pl.col(col).fill_null("").str.split(",")).explode(col).filter(pl.col(col) != "").rename({col: "e"})  # noqa: E731
mm, cc = ex(m, "matched_entity_ids"), ex(c, "candidate_entity_ids")
miss = mm.join(cc, on=["source1_entity_id", "e"], how="anti")
print(f"matches missing from candidates: {miss.height} over {miss['source1_entity_id'].n_unique()} S1s")
u = pl.concat([cc, miss]).group_by("source1_entity_id", maintain_order=True).agg(pl.col("e").str.join(",").alias("candidate_entity_ids"))
o = c.select("source1_entity_id").with_row_index("ord").join(u, on="source1_entity_id", how="left").sort("ord") \
     .with_columns(pl.col("candidate_entity_ids").fill_null("")).select("source1_entity_id", "candidate_entity_ids")
with open(out / "candidate_pairs.tsv", "w", encoding="utf-8", newline="\n") as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")
    for s, e in o.iter_rows():
        f.write(f"{s}\t{e}\n")
print("rewrote", out / "candidate_pairs.tsv", o.height, "rows")

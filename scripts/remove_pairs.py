"""Remove listed (record, S1) pairs from a submission (candidate file unchanged: removed pairs stay scored candidates).

uv run --no-project --with polars==1.44.2 python scripts/remove_pairs.py <base_dir> <out_dir> <pairs.parquet> [<pairs2.parquet> ...]
"""
import sys
from pathlib import Path

import polars as pl

base, out = Path(sys.argv[1]), Path(sys.argv[2])
rm = pl.concat([pl.read_parquet(p).select("m", "s1") for p in sys.argv[3:]]).unique()
drop = set(zip(rm["m"].to_list(), rm["s1"].to_list()))
g = pl.read_csv(base / "matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
n = 0
out.mkdir(parents=True, exist_ok=True)
with open(out / "matching_results.tsv", "w", encoding="utf-8", newline="\n") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for s, m in zip(g["source1_entity_id"].to_list(), g["matched_entity_ids"].fill_null("").to_list()):
        keep = []
        for e in (m.split(",") if m else []):
            if (e, s) in drop:
                n += 1
            else:
                keep.append(e)
        f.write(f"{s}\t{','.join(keep)}\n")
(out / "candidate_pairs.tsv").write_bytes((base / "candidate_pairs.tsv").read_bytes())
print(f"removed {n} of {len(drop)} listed pairs; wrote {out}")

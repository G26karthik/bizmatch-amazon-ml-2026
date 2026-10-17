"""E01 step 2: inspect unaccepted strong-name gap pairs (add candidates) and the suffix-collapse side (12 vs 12 bis/ter),
write /work/agent_e01/fr_parse_adds.parquet, fr_parse_conflicts.parquet, fr_parse_removals.parquet.

python scripts/e01_adds.py   (after scripts/e01_gap.py)
"""
import sys
from collections import Counter
from pathlib import Path

import polars as pl

sys.path.insert(0, "/work/agent_e01")
from e01_gap_lib import DECOY, canon, name_rel, toks  # noqa: E402

OUT = Path("/work/agent_e01")
pl.Config.set_fmt_str_lengths(60); pl.Config.set_tbl_rows(120); pl.Config.set_tbl_width_chars(300)
gp = pl.read_parquet(OUT / "gap_pairs.parquet")
ctrl = pl.read_parquet(OUT / "ctrl_pairs.parquet")
STRONG = ["exact", "case", "punct", "legal", "reorder"]
SHOW = ["q", "s1", "business_name_q", "business_name_s", "addr_n_q", "addr_n_s", "name_rel", "status", "p3", "p7", "pfx",
        "best7", "bestfx", "s1_other_acc", "cap_ok", "decoy_word"]

print("=== unaccepted gap pairs by name relation (unassigned / other S1)")
print(gp.filter(~pl.col("acc")).group_by("rel_kind", "status").agg(pl.len(), pl.col("p3").median(), pl.col("p7").median()).sort("len", descending=True))
cand = gp.filter(~pl.col("acc") & pl.col("rel_kind").is_in(STRONG + ["typo", "add1", "drop1"]))
print(cand.sort("rel_kind", "status").select(SHOW))

# the same for control pairs: are the unaccepted strong-name control pairs similar (then the cause is not the gap)?
cc = ctrl.filter(~pl.col("acc") & pl.col("rel_kind").is_in(STRONG))
print("\n=== control (no gap) unaccepted strong-name pairs, sample")
print(cc.group_by("status").agg(pl.len(), pl.col("p3").median(), pl.col("p7").median(), (pl.col("best7") == pl.col("s1")).mean().alias("is_best7")))
print(cc.select(SHOW).head(15))

# how many records of the unaccepted strong-name gap pairs have ANOTHER candidate S1 at the gap-equal address (competition)?
comp = gp.group_by("q").agg(pl.col("s1").n_unique().alias("n_gap_s1"))
cand = cand.join(comp, on="q", how="left")
print(cand.group_by("n_gap_s1").len())

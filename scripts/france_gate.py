"""Gated France update: keep the base decisions and add only those France pairs that the France-cleaned blend selects
AND that carry the true-copy signature (same house number, street similarity >= 90). France has no labels; the
signature is what separates true copies (same house) from generator decoys (same street, nearby house number).

python scripts/france_gate.py --work /work/run20 --base_tag run20_both --clean_tag run20_frce --pruned /work/frfix/pruned --out /work/out_frgate     [--rule /work/run20/testlike_rule_run20_both.json]
"""
import argparse
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, "src")
from ber.pipeline import decide_pairs  # noqa: E402
from ber.stage2 import _chosen, _write_outputs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--work", type=Path, required=True)
ap.add_argument("--base_tag", required=True)
ap.add_argument("--clean_tag", required=True)
ap.add_argument("--pruned", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--rule", type=Path, help="test-like rule json (ge, ga) from scripts/testlike_rule.py; default: production rule")
a = ap.parse_args()
cfg = json.loads((a.work / f"val_report_{a.base_tag}.json").read_text())["best"]
rec = pl.read_parquet(a.work / "test" / "records.parquet", columns=["entity_id", "src", "country", "addr_empty"]).with_row_index("rid")
emp = rec.select(pl.col("rid").cast(pl.UInt32).alias("q"), "addr_empty")


def decide(best: pl.DataFrame) -> pl.DataFrame:
    if a.rule is None:
        return _chosen(best, cfg)
    r = json.loads(a.rule.read_text())
    b = best.with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32)).join(emp, on="q", how="left")
    g = pl.when(pl.col("addr_empty").fill_null(False)).then(r["ge"]).otherwise(r["ga"])
    return decide_pairs(b.with_columns(pl.col("prob").pow(g).alias("prob")).select("q", "s1", "prob"), 0.0)


base = decide(pl.read_parquet(a.work / f"test_best_{a.base_tag}.parquet"))
clean = decide(pl.read_parquet(a.work / f"test_best_{a.clean_tag}.parquet"))
fr = rec.filter(pl.col("country") == "France").select(pl.col("rid").cast(pl.UInt32).alias("s1"))
feat = pl.concat([pl.read_parquet(p, columns=["q", "s1", "house_eq", "street_tsr"]) for p in sorted(a.pruned.glob("*.parquet"))]) \
         .with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
add = clean.join(base, on=["q", "s1"], how="anti").join(fr, on="s1", how="semi").join(feat, on=["q", "s1"], how="left")
gated = add.filter(pl.col("house_eq").fill_null(False) & (pl.col("street_tsr").fill_null(0) >= 90)).select("q", "s1")
# a record may be added to an S1 only if the base decision does not already give that record to another S1
gated = gated.join(base.select("q"), on="q", how="anti")
print(f"cleaned blend adds {add.height} France pairs; with the true-copy signature {gated.height}")
final = pl.concat([base, gated])
cands = pl.read_parquet(a.work / "scores_test.parquet", columns=["q", "s1"])
b = final.with_columns(pl.lit(1.0, dtype=pl.Float32).alias("prob"))
_write_outputs(a.work, a.out, {"mode": "thr", "thr": 0.5}, b, cands, rec.select("rid", "entity_id", "src"))
print(f"final pairs {final.height} (base {base.height})")

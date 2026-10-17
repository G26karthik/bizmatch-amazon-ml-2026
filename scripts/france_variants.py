"""France-only variants of a blended submission (US / India decisions unchanged).

France has no labels, so these are judged on the leaderboard. Evidence (scripts/france_fix.py):
French noise words ("developpement", "groupe", "associes", "fils") that replace a word of the S1 name mark true copies
far more often than decoys when the record keeps the S1's street and house number (decoys: same street, other house
number, like US "Midtown" branches), but the cross-encoders (trained on US/India) read them as different businesses.

  A (rule):  France S1s re-decided with a less strict rule (gamma 1.0 instead of the validated 1.5).
  B (boost): France records whose name contains one of those noise words that the S1 name lacks, with equal house number
             and street similarity >= 90, get prob = max(prob, BOOST) before the usual decision; decoy words untouched.

python scripts/france_variants.py --work /work/run20 --tag run20_large --pruned /work/frfix/pruned --out /work/out_fr
"""
import argparse
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, "src")
from ber.stage2 import _chosen, _write_outputs  # noqa: E402

NOISE = ["developpement", "groupe", "associes", "fils"]
BOOST = 0.9

ap = argparse.ArgumentParser()
ap.add_argument("--work", type=Path, required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--pruned", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
cfg = json.loads((a.work / f"val_report_{a.tag}.json").read_text())["best"]
rec = pl.read_parquet(a.work / "test" / "records.parquet", columns=["entity_id", "src", "country", "name_n"]).with_row_index("rid")
best = pl.read_parquet(a.work / f"test_best_{a.tag}.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
cands = pl.read_parquet(a.work / "scores_test.parquet", columns=["q", "s1"])
fr = rec.filter(pl.col("country") == "France").select(pl.col("rid").cast(pl.UInt32).alias("s1"))
is_fr = best["s1"].is_in(fr["s1"].implode())
base = _chosen(best, cfg)


def write(name: str, chosen: pl.DataFrame) -> None:
    d = base.join(chosen, on=["q", "s1"], how="anti").height, chosen.join(base, on=["q", "s1"], how="anti").height
    print(f"{name}: {chosen.height} pairs; vs base: removed {d[0]}, added {d[1]}")
    # _write_outputs applies cfg to `best`; pass the chosen pairs directly via a threshold-0 config
    b = chosen.with_columns(pl.lit(1.0, dtype=pl.Float32).alias("prob"))
    _write_outputs(a.work, a.out / name, {"mode": "thr", "thr": 0.5}, b, cands, rec.select("rid", "entity_id", "src"))


write("base", base)  # must reproduce the submitted file exactly

# A: France with gamma 1.0
ca = pl.concat([_chosen(best.filter(~is_fr), cfg), _chosen(best.filter(is_fr), dict(cfg, gamma=1.0))])
write("A_fr_gamma1", ca)

# B: boost noise-word true-copy signature in France
fa = pl.concat([pl.read_parquet(p, columns=["q", "s1", "house_eq", "street_tsr"]) for p in sorted(a.pruned.glob("*.parquet"))]) \
       .with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
nm = rec.select(pl.col("rid").cast(pl.UInt32), "name_n")
bb = best.filter(is_fr).join(fa, on=["q", "s1"], how="left") \
         .join(nm.rename({"rid": "q", "name_n": "qn"}), on="q").join(nm.rename({"rid": "s1", "name_n": "sn"}), on="s1")
word = pl.lit(False)
for w in NOISE:
    word = word | (pl.col("qn").str.contains(rf"\b{w}\b") & ~pl.col("sn").str.contains(rf"\b{w}\b"))
hit = word & pl.col("house_eq").fill_null(False) & (pl.col("street_tsr").fill_null(0) >= 90)
bb = bb.with_columns(hit.alias("hit"))
print(f"B: France records matching the signature: {bb['hit'].sum()} (currently selected {bb.filter(pl.col('hit')).join(base, on=['q', 's1'], how='semi').height})")
boosted = bb.with_columns(pl.when(pl.col("hit")).then(pl.max_horizontal("prob", pl.lit(BOOST))).otherwise(pl.col("prob")).alias("prob")) \
            .select("q", "s1", pl.col("prob").cast(pl.Float32))
cb = pl.concat([_chosen(best.filter(~is_fr), cfg), _chosen(boosted, cfg)])
write("B_fr_boost", cb)

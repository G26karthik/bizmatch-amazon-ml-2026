"""Build the teammate validation kit: val_split.csv.gz (entity_id, role) matching our validation protocol, plus our
own validation predictions (half A and B) in submission format to check the scorer reproduces our numbers.

Roles: valA_s1 / valB_s1 (validation S1 entities, by half), hidden_s1 (19% of training S1 entities removed from the
pool so training matches the test density), val_record (S2/S3 records of validation groups: never train on them).

python scripts/build_val_kit.py --data /work/data --base /work/base --work /work/run13 --tag sbw --out /work/val_kit
"""
import argparse
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, "src")
from ber.pipeline import SEED, hide_pct, hidden_s1, labels  # noqa: E402
from ber.stage2 import _chosen  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, default=Path("/work/data"))
ap.add_argument("--base", type=Path, default=Path("/work/base"))
ap.add_argument("--work", type=Path, required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)

lab = labels(a.data, a.base).with_columns(
    (pl.col("parent").fill_null(pl.col("rid")).hash(SEED + 7) % 2 == 0).alias("half_a"))
rec = pl.read_parquet(a.base / "train" / "records.parquet", columns=["entity_id", "src"])
hid = rec.filter((pl.col("src") == 1) & hidden_s1(pl.col("entity_id"), hide_pct(a.base))).select(
    "entity_id", pl.lit("hidden_s1").alias("role"))
s1v = lab.filter((pl.col("src") == 1) & pl.col("is_val")).select(
    "entity_id", pl.when(pl.col("half_a")).then(pl.lit("valA_s1")).otherwise(pl.lit("valB_s1")).alias("role"))
vrec = lab.filter((pl.col("src") != 1) & pl.col("is_val")).select("entity_id", pl.lit("val_record").alias("role"))
split = pl.concat([s1v, hid, vrec])
split.write_csv(a.out / "val_split.csv")
print(split.group_by("role").len().sort("role"))

# our validation predictions in submission format (rule chosen on half A, as in the blend report)
cfg = json.loads((a.work / f"val_report_{a.tag}.json").read_text())["best"]
ch = _chosen(pl.read_parquet(a.work / f"val_best_{a.tag}.parquet"), cfg)
ids = lab.select(pl.col("rid").cast(pl.UInt32), "entity_id")
ch = ch.with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32)) \
       .join(ids.rename({"rid": "q", "entity_id": "qid"}), on="q").join(ids.rename({"rid": "s1", "entity_id": "sid"}), on="s1")
pm = ch.group_by("sid").agg(pl.col("qid").sort().str.join(",").alias("matched_entity_ids"))
out = s1v.select(pl.col("entity_id").alias("sid")).join(pm, on="sid", how="left").select(
    pl.col("sid").alias("source1_entity_id"), pl.col("matched_entity_ids").fill_null(""))
out.write_csv(a.out / f"our_val_predictions_{a.tag}.tsv", separator="\t", quote_style="never")
print("wrote", out.height, "rows of our validation predictions; rule", cfg)

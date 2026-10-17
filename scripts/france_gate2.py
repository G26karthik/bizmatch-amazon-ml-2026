"""France gate v2: add unassigned France records whose best candidate S1 sits at the same address (house number equal,
street similarity >= 90 on France-cleaned features) and whose name keeps a distinctive word of the S1 name.

Evidence (26 Sep night): copies get one noise operation; decoys always move the house number; S1 businesses that share
an address have unrelated names (4.7% share a word); in US/India validation, same-address pairs with low name
similarity are 85-94% true and accepted accordingly, while France accepts ~70% (type-word swaps such as
"X Club" -> "X Anciens"). Excluded: records adding a decoy word, S1s already at the per-source copy cap.

python france_gate2.py --base_out /work/out_run28_6ce_pgate --work /work/run28 --tag run28_6ce --out /work/out_run28_6ce_pg2
"""
import argparse
import glob
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import polars as pl

sys.path.insert(0, "src")
from ber.stage2 import _write_outputs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base_out", type=Path, required=True)
ap.add_argument("--work", type=Path, required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--out", type=Path)
ap.add_argument("--max_df", type=float, default=0.002, help="a shared word counts if at most this share of France S1 names has it")
a = ap.parse_args()

rec = pl.read_parquet(a.work / "test" / "records.parquet", columns=["entity_id", "src", "country", "business_name"]).with_row_index("rid") \
        .with_columns(pl.col("rid").cast(pl.UInt32))
eid = rec.select("rid", "entity_id")
base = pl.read_csv(a.base_out / "matching_results.tsv", separator="\t")
pairs = base.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls("matched_entity_ids") \
            .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
            .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1")
assigned = pairs.select("q").unique()
best = pl.read_parquet(a.work / f"test_best_{a.tag}.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
feat = pl.concat([pl.read_parquet(p, columns=["q", "s1", "street_tsr", "house_eq"]) for p in sorted(glob.glob("/work/frfix/pruned/France*.parquet"))]) \
         .with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
c = best.join(feat, on=["q", "s1"], how="inner").filter(pl.col("house_eq").fill_null(False) & (pl.col("street_tsr") >= 90)) \
        .join(assigned, on="q", how="anti") \
        .join(rec.select(pl.col("rid").alias("q"), pl.col("business_name").alias("nq"), "src"), on="q") \
        .join(rec.select(pl.col("rid").alias("s1"), pl.col("business_name").alias("ns")), on="s1")

STOP = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "de", "des", "du", "la", "le", "les", "et", "d", "l", "s", "a", "u", "r", "e"}
DECOY = {"holding", "participations", "international", "distribution", "groupe", "developpement", "france"}


def toks(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return set(re.findall(r"[a-z0-9]+", s)) - STOP


s1names = rec.filter((pl.col("src") == 1) & (pl.col("country") == "France"))["business_name"].to_list()
df = Counter(t for n in s1names for t in toks(n))
cut = a.max_df * len(s1names)
keep, why = [], Counter()
for nq, ns in zip(c["nq"].to_list(), c["ns"].to_list()):
    tq, ts = toks(nq), toks(ns)
    if (tq - ts) & DECOY:
        keep.append(False); why["decoy word"] += 1; continue
    shared = {t for t in tq & ts if df.get(t, 0) <= cut}
    if not shared:
        keep.append(False); why["no distinctive shared word"] += 1; continue
    keep.append(True); why["kept"] += 1
c = c.with_columns(pl.Series("keep", keep))
print(f"France unassigned same-address best pairs: {c.height}; {dict(why)}")
# per-source copy caps (<= 5 S2, <= 6 S3 per S1) counting the base decisions
cnt = pairs.join(rec.select(pl.col("rid").alias("q"), "src"), on="q").group_by("s1").agg((pl.col("src") == 2).sum().alias("n2"), (pl.col("src") == 3).sum().alias("n3"))
k = c.filter("keep").join(cnt, on="s1", how="left").with_columns(pl.col("n2").fill_null(0), pl.col("n3").fill_null(0))
k = k.with_columns(pl.when(pl.col("src") == 2).then(pl.col("n2") < 5).otherwise(pl.col("n3") < 6).alias("room"))
add = k.filter("room").select("q", "s1")
print(f"kept {k.height}, with room under the copy cap {add.height}; prob of added pairs:",
      k.filter("room").select(pl.col("prob").cut([0.01, 0.05, 0.2, 0.5]).value_counts(sort=True)).unnest("prob").rows())
print(k.filter("room").sample(n=min(20, add.height), seed=5).select(pl.col("prob").round(3), "nq", "ns"))
if a.out:
    final = pl.concat([pairs, add])
    cands = pl.read_parquet(a.work / "scores_test.parquet", columns=["q", "s1"])
    _write_outputs(a.work, a.out, {"mode": "thr", "thr": 0.5}, final.with_columns(pl.lit(1.0, dtype=pl.Float32).alias("prob")),
                   cands, rec.select("rid", "entity_id", "src"))
    print(f"final pairs {final.height} (base {pairs.height})")

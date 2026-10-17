"""Exact-address rescue: records that no S1 claims but whose normalised address (with house number) equals exactly one S1's
address in the pool. Generator decoys always move the house number, so an exact address match excludes them; copies whose
name was replaced (pseudo brands such as "Fluxcalonovi", initials such as "AS", garbled names) are missed by retrieval and
stage-A pruning because their name similarity is zero (validation error audit: 44% of the remaining loss is true S1s that
never became candidates).

Validation mode (default): evaluates the rule on half A / half B (standard metric, and test-like: copies of hidden S1s
removed), for all names and for pseudo/initial names only. Test mode (--test): writes the added pairs.

python scripts/exact_addr_rule.py --work /work/run28 --tag run28_7q [--test --base_out /work/out_fr4 --out /work/out_fr4x]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "src")
from ber.metric import entity_f05  # noqa: E402
from ber.pipeline import SEED, hidden_s1, labels  # noqa: E402
from ber.stage2 import _chosen, _write_outputs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--work", type=Path, default=Path("/work/run28"))
ap.add_argument("--tag", default="run28_7q")
ap.add_argument("--test", action="store_true")
ap.add_argument("--base_out", type=Path)
ap.add_argument("--out", type=Path)
ap.add_argument("--records", type=Path, help="test records (France-fixed by default)", default=Path("/work/runfr/test/records.parquet"))
a = ap.parse_args()

SYL = "riza nyla orbi aria lyra mira gild brix tavo vantage umbra zeta xylo zeph yuma drex delta cira kelo dova onyx vera halo ecto pyra flux calo novi belo".split()
PSEUDO = re.compile(r"^(?:" + "|".join(SYL) + r"){2,}\b", re.I)


def is_pseudo(name):
    n = (name or "").strip()
    first = n.split()[0].lower() if n.split() else ""
    return bool(PSEUDO.match(first)) or (len(re.sub(r"[^A-Za-z]", "", n)) <= 3 and n.isupper())


def exact_pairs(rec, pool_mask):
    """(q, s1) where q is S2/S3 and s1 the unique pool S1 with the same country + normalised address (house present)."""
    r = rec.filter(~pl.col("addr_empty").fill_null(False) & pl.col("house").is_not_null() & (pl.col("house") != "") & (pl.col("addr_n") != ""))
    s1 = r.filter((pl.col("src") == 1) & pool_mask(r)).select(pl.col("rid").alias("s1"), "country", "addr_n")
    cnt = s1.group_by("country", "addr_n").agg(pl.len().alias("n_s1"), pl.col("s1").first())
    q = r.filter(pl.col("src") != 1).select(pl.col("rid").alias("q"), "country", "addr_n", "business_name")
    return q.join(cnt.filter(pl.col("n_s1") == 1), on=["country", "addr_n"]).select("q", "s1", "business_name")


if not a.test:
    rec = pl.read_parquet(a.work / "train" / "records.parquet", columns=["entity_id", "src", "country", "addr_n", "house", "addr_empty", "business_name"]) \
            .with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
    lab = labels(Path("/work/data"), a.work).with_columns((pl.col("parent").fill_null(pl.col("rid")).hash(SEED + 7) % 2 == 0).alias("half_a"))
    hid = set(rec.filter((pl.col("src") == 1) & hidden_s1(pl.col("entity_id"), 19))["rid"].to_list())
    pairs = exact_pairs(rec, lambda r: ~pl.col("rid").is_in(list(hid)))
    s1v = lab.filter((pl.col("src") == 1) & pl.col("is_val"))
    S = {h: s1v.filter(pl.col("half_a") == (h == "A"))["rid"].to_list() for h in "AB"}
    valset = set(S["A"]) | set(S["B"])
    parent = dict(lab.filter(pl.col("parent").is_not_null()).select("rid", "parent").iter_rows())
    truth = {}
    for q, p in parent.items():
        truth.setdefault(p, set()).add(q)
    gt = pl.read_parquet("/work/data/train_pairs.parquet")
    hidE = rec.filter((pl.col("src") == 1) & hidden_s1(pl.col("entity_id"), 19)).select(pl.col("entity_id").alias("s1"))
    H = set(gt.join(hidE, on="s1", how="semi").select(pl.col("m").alias("entity_id")).join(rec.select("entity_id", "rid"), on="entity_id")["rid"].to_list())
    cfg = json.loads((a.work / f"val_report_{a.tag}.json").read_text())["best"]
    ch = _chosen(pl.read_parquet(a.work / f"val_best_{a.tag}.parquet"), cfg).select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
    cap = ch.join(rec.select(pl.col("rid").alias("q"), "src"), on="q").group_by("s1").agg((pl.col("src") == 2).sum().alias("n2"), (pl.col("src") == 3).sum().alias("n3"))
    cand = pairs.filter(pl.col("s1").is_in(list(valset))).join(ch.select("q"), on="q", how="anti") \
                .join(rec.select(pl.col("rid").alias("q"), "src"), on="q").join(cap, on="s1", how="left") \
                .with_columns(pl.col("n2").fill_null(0), pl.col("n3").fill_null(0)) \
                .filter(pl.when(pl.col("src") == 2).then(pl.col("n2") < 5).otherwise(pl.col("n3") < 6))
    cand = cand.with_columns(pl.Series("pseudo", [is_pseudo(n) for n in cand["business_name"].to_list()]),
                             pl.Series("true", [parent.get(q) == s for q, s in zip(cand["q"].to_list(), cand["s1"].to_list())]),
                             pl.col("q").is_in(list(H)).alias("H"))

    def macro(chs, half, drop_h=False):
        pm = {}
        for q, s in chs.iter_rows():
            if drop_h and q in H:
                continue
            pm.setdefault(s, set()).add(q)
        return float(np.mean([entity_f05(truth.get(s, ()), pm.get(s, ())) for s in S[half]]))

    base = {(h, t): macro(ch, h, t) for h in "AB" for t in (False, True)}
    print(f"exact-address candidates for unassigned records (val S1s, under cap): {cand.height}; true {cand['true'].sum()}; "
          f"copies of hidden S1s {cand['H'].sum()}; pseudo {cand['pseudo'].sum()}")
    for name, sub in (("all names", cand), ("pseudo/initials only", cand.filter("pseudo")), ("non-pseudo", cand.filter(~pl.col("pseudo")))):
        new = pl.concat([ch, sub.select("q", "s1")])
        prec = sub["true"].mean() if sub.height else 0.0
        prec_tl = sub.filter(~pl.col("H"))["true"].mean() if sub.filter(~pl.col("H")).height else 0.0
        g = {(h, t): macro(new, h, t) - base[(h, t)] for h in "AB" for t in (False, True)}
        print(f"RULE {name}: adds {sub.height} (precision {prec:.3f}; without hidden-S1 copies {prec_tl:.3f}) "
              f"standard A {g[('A', False)]:+.6f} B {g[('B', False)]:+.6f} | test-like A {g[('A', True)]:+.6f} B {g[('B', True)]:+.6f}")
else:
    rec = pl.read_parquet(a.records, columns=["entity_id", "src", "country", "addr_n", "house", "business_name"]).with_row_index("rid") \
            .with_columns(pl.col("rid").cast(pl.UInt32))
    rec = rec.join(pl.read_parquet(a.work / "test" / "records.parquet", columns=["addr_empty"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32)), on="rid")
    pairs = exact_pairs(rec, lambda r: pl.lit(True))
    eid = rec.select("rid", "entity_id")
    g = pl.read_csv(a.base_out / "matching_results.tsv", separator="\t")
    cur = g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls("matched_entity_ids") \
           .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
           .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1")
    cap = cur.join(rec.select(pl.col("rid").alias("q"), "src"), on="q").group_by("s1").agg((pl.col("src") == 2).sum().alias("n2"), (pl.col("src") == 3).sum().alias("n3"))
    add = pairs.join(cur.select("q"), on="q", how="anti").join(rec.select(pl.col("rid").alias("q"), "src"), on="q") \
               .join(cap, on="s1", how="left").with_columns(pl.col("n2").fill_null(0), pl.col("n3").fill_null(0)) \
               .filter(pl.when(pl.col("src") == 2).then(pl.col("n2") < 5).otherwise(pl.col("n3") < 6))
    add = add.with_columns(pl.Series("pseudo", [is_pseudo(n) for n in add["business_name"].to_list()])) \
             .join(rec.select(pl.col("rid").alias("s1"), "country"), on="s1")
    print(f"test exact-address additions: {add.height}; by country {add.group_by('country').len().rows()}; pseudo {add['pseudo'].sum()}")
    if a.out:
        final = pl.concat([cur, add.select("q", "s1")])
        cands = pl.read_parquet(a.work / "scores_test.parquet", columns=["q", "s1"])
        _write_outputs(a.work, a.out, {"mode": "thr", "thr": 0.5}, final.with_columns(pl.lit(1.0, dtype=pl.Float32).alias("prob")),
                       cands, rec.select("rid", "entity_id", "src"))
        print(f"final pairs {final.height}")

"""Unique-name empty-address rule: an unassigned record with an empty address whose name key (token set, legal words
removed) equals exactly one S1's name key in the pool (no namesakes) goes to that S1. Validation mode reports the half A /
half B effect (standard and test-like), per variant; --test writes the additions for a test output.

python scripts/unique_empty_rule.py [--test --base_out /work/out_x --out /work/out_y]
"""
import argparse
import json
import re
import sys
import unicodedata
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
ap.add_argument("--min_prob", type=float, default=0.0)
a = ap.parse_args()
LEGAL = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "inc", "llc", "ltd", "pvt", "private", "limited", "corp", "corporation",
         "incorporated", "company", "co", "lp", "llp", "pllc", "pc", "plc", "the", "de", "des", "du", "la", "le", "les", "et", "of", "and"}


def nkey(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return " ".join(sorted(set(re.findall(r"[a-z0-9]+", s)) - LEGAL))


def rule_pairs(rec, pool_mask, best):
    s1 = rec.filter((pl.col("src") == 1) & pool_mask).select(pl.col("rid").alias("s1"), "country", "nk")
    cnt = s1.group_by("country", "nk").agg(pl.len().alias("n"), pl.col("s1").first())
    q = rec.filter((pl.col("src") != 1) & pl.col("addr_empty").fill_null(False) & (pl.col("nk").str.len_chars() > 0)) \
           .select(pl.col("rid").alias("q"), "country", "nk")
    x = q.join(cnt.filter(pl.col("n") == 1), on=["country", "nk"]).select("q", "s1")
    return x.join(best.select("q", "s1", "prob"), on=["q", "s1"], how="left").with_columns(pl.col("prob").fill_null(0.0))


if not a.test:
    rec = pl.read_parquet(a.work / "train" / "records.parquet", columns=["entity_id", "src", "country", "business_name", "addr_empty"]) \
            .with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
    rec = rec.with_columns(pl.Series("nk", [nkey(n) for n in rec["business_name"].to_list()]))
    hid = rec.filter((pl.col("src") == 1) & hidden_s1(pl.col("entity_id"), 19))["rid"]
    lab = labels(Path("/work/data"), a.work).with_columns((pl.col("parent").fill_null(pl.col("rid")).hash(SEED + 7) % 2 == 0).alias("half_a"))
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
    vb = pl.read_parquet(a.work / f"val_best_{a.tag}.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
    ch = _chosen(vb, cfg).select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
    cand = rule_pairs(rec, ~pl.col("rid").is_in(hid.implode()), vb).filter(pl.col("s1").is_in(list(valset))).join(ch.select("q"), on="q", how="anti")
    cand = cand.with_columns(pl.Series("true", [parent.get(q) == s for q, s in zip(cand["q"].to_list(), cand["s1"].to_list())]),
                             pl.col("q").is_in(list(H)).alias("H"))

    def macro(chs, half, drop_h=False):
        pm = {}
        for q, s in chs.iter_rows():
            if drop_h and q in H:
                continue
            pm.setdefault(s, set()).add(q)
        return float(np.mean([entity_f05(truth.get(s, ()), pm.get(s, ())) for s in S[half]]))

    base = {(h, t): macro(ch, h, t) for h in "AB" for t in (False, True)}
    for mp in (0.0, 0.05, 0.1, 0.2, 0.3):
        sub = cand.filter(pl.col("prob") >= mp)
        new = pl.concat([ch, sub.select("q", "s1")])
        g = {(h, t): macro(new, h, t) - base[(h, t)] for h in "AB" for t in (False, True)}
        print(f"prob>={mp}: adds {sub.height} precision {sub['true'].mean() if sub.height else 0:.3f} (non-H {sub.filter(~pl.col('H'))['true'].mean() if sub.filter(~pl.col('H')).height else 0:.3f}) "
              f"standard A {g[('A', False)]:+.6f} B {g[('B', False)]:+.6f} | test-like A {g[('A', True)]:+.6f} B {g[('B', True)]:+.6f}")
else:
    rec = pl.read_parquet(a.work / "test" / "records.parquet", columns=["entity_id", "src", "country", "business_name", "addr_empty"]) \
            .with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
    rec = rec.with_columns(pl.Series("nk", [nkey(n) for n in rec["business_name"].to_list()]))
    tb = pl.read_parquet(a.work / f"test_best_{a.tag}.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
    eid = rec.select("rid", "entity_id")
    g = pl.read_csv(a.base_out / "matching_results.tsv", separator="\t")
    cur = g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls("matched_entity_ids") \
           .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
           .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1")
    add = rule_pairs(rec, pl.lit(True), tb).filter(pl.col("prob") >= a.min_prob).join(cur.select("q"), on="q", how="anti")
    add = add.join(rec.select(pl.col("rid").alias("s1"), "country"), on="s1")
    print(f"test additions: {add.height}; by country {add.group_by('country').len().rows()}")
    add.select("q", "s1", "country").write_parquet("/work/dive/unique_empty_test.parquet")

"""Ambiguous pseudo-name rule. An accepted record whose name shares no word with its S1 (after legal/stop words), sits
at the S1's exact address, and that address holds 2+ S1s, is assigned by a coin flip unless its name encodes one S1
(web handle / initials). Removal is right when the pick is < ~70% precise. Validation mode: precision of such accepted
pairs and the F effect of removing them on half A / half B (standard and test-like); --test: removal list per country.

python scripts/ambig_pseudo_rule.py [--test --base_out /work/out_fr8]
"""
import argparse
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "src")
from ber.metric import entity_f05  # noqa: E402
from ber.pipeline import SEED, hidden_s1, labels  # noqa: E402
from ber.stage2 import _chosen  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--work", type=Path, default=Path("/work/run28"))
ap.add_argument("--tag", default="run28_7q")
ap.add_argument("--test", action="store_true")
ap.add_argument("--base_out", type=Path)
a = ap.parse_args()
LEG = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "inc", "llc", "ltd", "pvt", "private", "limited", "corp", "corporation",
       "incorporated", "company", "co", "lp", "llp", "pllc", "pc", "plc", "the", "de", "des", "du", "la", "le", "les", "et", "of", "and",
       "s", "a", "d", "l", "com", "www", "dba", "aka", "fka", "ta"}


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


def prep(rec):
    return rec.with_columns(pl.Series("ak", [" ".join(sorted(set(re.findall(r"[a-z0-9]+", norm(x))))) if x else "" for x in rec["addr_n"].to_list()]),
                            pl.Series("nt", [sorted(set(re.findall(r"[a-z0-9]+", norm(n))) - LEG) for n in rec["business_name"].to_list()]),
                            pl.Series("flat", [re.sub(r"[^a-z0-9]", "", norm(n)) for n in rec["business_name"].to_list()]))


def encodes(qflat, s_nt):
    """record name encodes the S1 (web handle / concatenation / initials)."""
    if not s_nt:
        return False
    cat = "".join(sorted(s_nt))
    if any(w in qflat for w in s_nt if len(w) >= 4):
        return True
    ini = "".join(w[0] for w in s_nt)
    return len(qflat) <= 5 and set(qflat) <= set(ini + "".join(s_nt))


def flag(pairs, rec):
    """pairs (q, s1) -> subset that is ambiguous pseudo: no word overlap, same address, 2+ S1s there, name encodes none."""
    s1a = rec.filter((pl.col("src") == 1) & (pl.col("ak") != "")).group_by("country", "ak").agg(pl.col("rid").alias("here"))
    x = pairs.join(rec.select(pl.col("rid").alias("q"), pl.col("ak").alias("akq"), pl.col("nt").alias("ntq"), pl.col("flat").alias("flq")), on="q") \
             .join(rec.select(pl.col("rid").alias("s1"), "country", pl.col("ak").alias("aks"), pl.col("nt").alias("nts")), on="s1")
    x = x.filter((pl.col("akq") == pl.col("aks")) & (pl.col("aks") != ""))
    x = x.filter(pl.Series([not (set(p) & set(s)) for p, s in zip(x["ntq"].to_list(), x["nts"].to_list())]))
    x = x.join(s1a.rename({"ak": "aks"}), on=["country", "aks"]).filter(pl.col("here").list.len() >= 2)
    nts = dict(zip(rec["rid"].to_list(), rec["nt"].to_list()))
    keep = []
    for fl, s, here in zip(x["flq"].to_list(), x["s1"].to_list(), x["here"].to_list()):
        keep.append(not any(encodes(fl, nts[h]) for h in here))
    return x.filter(pl.Series(keep)).select("q", "s1", "country")


if not a.test:
    rec = prep(pl.read_parquet(a.work / "train" / "records.parquet", columns=["entity_id", "src", "country", "business_name", "addr_n"])
               .with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32)))
    lab = labels(Path("/work/data"), a.work).with_columns((pl.col("parent").fill_null(pl.col("rid")).hash(SEED + 7) % 2 == 0).alias("half_a"))
    s1v = lab.filter((pl.col("src") == 1) & pl.col("is_val"))
    S = {h: s1v.filter(pl.col("half_a") == (h == "A"))["rid"].to_list() for h in "AB"}
    parent = dict(lab.filter(pl.col("parent").is_not_null()).select("rid", "parent").iter_rows())
    truth = {}
    for q, p in parent.items():
        truth.setdefault(p, set()).add(q)
    gt = pl.read_parquet("/work/data/train_pairs.parquet")
    hidE = rec.filter((pl.col("src") == 1) & hidden_s1(pl.col("entity_id"), 19)).select(pl.col("entity_id").alias("s1"))
    H = set(gt.join(hidE, on="s1", how="semi").select(pl.col("m").alias("entity_id")).join(rec.select("entity_id", "rid"), on="entity_id")["rid"].to_list())
    import json
    cfg = json.loads((a.work / f"val_report_{a.tag}.json").read_text())["best"]
    vb = pl.read_parquet(a.work / f"val_best_{a.tag}.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
    ch = _chosen(vb, cfg).select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
    # hidden S1s are not in the pool: the address-S1 count must exclude them
    hid_rids = set(rec.filter((pl.col("src") == 1) & hidden_s1(pl.col("entity_id"), 19))["rid"].to_list())
    rec_pool = rec.filter(~pl.col("rid").is_in(list(hid_rids)))
    fl = flag(ch, rec_pool)
    fl = fl.with_columns(pl.Series("true", [parent.get(q) == s for q, s in zip(fl["q"].to_list(), fl["s1"].to_list())]),
                         pl.col("q").is_in(list(H)).alias("H"))
    print("flagged accepted val pairs:", fl.height, "precision", round(fl["true"].mean(), 4), "non-H precision",
          round(fl.filter(~pl.col("H"))["true"].mean(), 4), "by country", fl.group_by("country").agg(pl.len(), pl.col("true").mean().round(3)).rows())

    def macro(chs, half, drop_h=False):
        pm = {}
        for q, s in chs.iter_rows():
            if drop_h and q in H:
                continue
            pm.setdefault(s, set()).add(q)
        return float(np.mean([entity_f05(truth.get(s, ()), pm.get(s, ())) for s in S[half]]))

    new = ch.join(fl.select("q", "s1"), on=["q", "s1"], how="anti")
    for h in "AB":
        for t in (False, True):
            print(f"half {h} {'test-like' if t else 'standard'}: {macro(ch, h, t):.6f} -> {macro(new, h, t):.6f} ({macro(new, h, t) - macro(ch, h, t):+.6f})")
else:
    rec = prep(pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country", "business_name", "addr_n"])
               .with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32)))
    eid = rec.select("rid", "entity_id")
    g = pl.read_csv(a.base_out / "matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
    acc = g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls().filter(pl.col("matched_entity_ids") != "") \
           .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
           .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1")
    fl = flag(acc, rec)
    print("flagged test pairs by country:", fl.group_by("country").len().sort("country").rows())
    fl.write_parquet("/work/dive/ambig_pseudo_test.parquet")
    nm = dict(zip(rec["rid"].to_list(), rec["business_name"].to_list()))
    for q, s, c in fl.filter(pl.col("country") == "France").sample(n=12, seed=1).iter_rows():
        print(f"  REC {nm[q]}  ->  S1 {nm[s]}")

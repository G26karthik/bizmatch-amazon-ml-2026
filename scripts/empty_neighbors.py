"""Predicted-empty S1s: do they have copy-looking records elsewhere? For each country, S1s our output leaves empty, and
the records that share their exact name key (anywhere) or their exact address (token set), with where those records
went (unassigned / assigned to another S1). US (test F = validation) calibrates what true singletons look like.

python scripts/empty_neighbors.py /work/out_fr8b
"""
import random
import re
import sys
import unicodedata
from collections import Counter

import polars as pl

out = sys.argv[1]
LEG = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "inc", "llc", "ltd", "pvt", "private", "limited", "corp", "corporation",
       "incorporated", "company", "co", "lp", "llp", "pllc", "pc", "plc", "the", "de", "des", "du", "la", "le", "les", "et", "of", "and", "s", "a", "d", "l"}


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country", "business_name", "business_address", "addr_n"]) \
        .with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
nk = [" ".join(sorted(set(re.findall(r"[a-z0-9]+", norm(n))) - LEG)) for n in rec["business_name"].to_list()]
ak = [" ".join(sorted(set(re.findall(r"[a-z0-9]+", norm(a))))) if a else "" for a in rec["addr_n"].to_list()]
eid = rec.select("rid", "entity_id")
g = pl.read_csv(f"{out}/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls().filter(pl.col("matched_entity_ids") != "") \
       .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
       .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1")
assigned = dict(zip(acc["q"].to_list(), acc["s1"].to_list()))
has = set(acc["s1"].to_list())
src = rec["src"].to_list(); ctry = rec["country"].to_list(); names = rec["business_name"].to_list(); raw = rec["business_address"].to_list()
random.seed(2)
for country in ("US", "France"):
    idx = [i for i in range(len(src)) if ctry[i] == country]
    by_nk, by_ak = {}, {}
    for i in idx:
        if src[i] != 1:
            if nk[i]:
                by_nk.setdefault(nk[i], []).append(i)
            if ak[i]:
                by_ak.setdefault(ak[i], []).append(i)
    empties = [i for i in idx if src[i] == 1 and i not in has]
    c = Counter(); ex = []
    for s in empties:
        same_name = by_nk.get(nk[s], []) if nk[s] else []
        same_addr = by_ak.get(ak[s], []) if ak[s] else []
        both = [q for q in same_name if ak[q] == ak[s]]
        tag = []
        if both:
            tag.append("name+addr record " + ("unassigned" if any(q not in assigned for q in both) else "assigned_elsewhere"))
        elif same_name:
            tag.append("name-only record " + ("unassigned" if any(q not in assigned for q in same_name) else "assigned_elsewhere"))
        if same_addr and not both:
            tag.append("addr-only record " + ("unassigned" if any(q not in assigned for q in same_addr) else "assigned_elsewhere"))
        key = " & ".join(tag) if tag else "nothing"
        c[key] += 1
        if len(ex) < 16 and tag and random.random() < 0.05:
            ex.append((s, both or same_name[:2] or same_addr[:2]))
    print(f"\n=== {country}: predicted-empty S1s {len(empties)}")
    for k, v in c.most_common(12):
        print(f"  {v:7d}  {v / len(empties):.3f}  {k}")
    if country == "France":
        for s, qs in ex:
            print(f"  S1  {names[s]} | {raw[s]}")
            for q in qs[:2]:
                a = assigned.get(q)
                print(f"      REC {names[q]} | {raw[q]}  -> " + (f"assigned to: {names[a]} | {raw[a]}" if a is not None else "unassigned"))

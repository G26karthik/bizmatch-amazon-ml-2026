"""Accepted France pairs whose record contains a decoy word: split by word and by whether the S1 name already has it
(added word = decoy signature; groupe / developpement are also generic copy-swap words).

python scripts/accepted_dw.py /work/out_fr12
"""
import re
import sys
import unicodedata
from collections import Counter

import polars as pl

out = sys.argv[1]
DWF = ["holding", "participations", "distribution", "international", "groupe", "developpement", "snc"]


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country", "business_name", "addr_n", "house"])
rec = rec.with_row_index("rid")
fr = rec.filter(pl.col("country") == "France")
tok = {e: set(re.findall(r"[a-z0-9]+", norm(n))) for e, n in zip(fr["entity_id"].to_list(), fr["business_name"].to_list())}
nm = dict(zip(fr["entity_id"].to_list(), fr["business_name"].to_list()))
ad = dict(zip(fr["entity_id"].to_list(), fr["addr_n"].to_list()))
g = pl.read_csv(f"{out}/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls() \
       .filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "q"})
acc = acc.filter(pl.col("s1").is_in(list(tok.keys())))
nacc = acc.group_by("s1").len().rename({"len": "n_acc"})
acc = acc.join(nacc, on="s1")
c = Counter(); ex = {}
rows = []
for s, q, n in acc.select("s1", "q", "n_acc").iter_rows():
    A, B = tok[q], tok[s]
    for w in DWF:
        if w in A:
            k = (w, "s1_has" if w in B else "added", "same_addr" if ad[q] == ad[s] else "diff_addr", "single" if n == 1 else "multi")
            c[k] += 1
            ex.setdefault(k, []).append((nm[s], nm[q], ad[s], ad[q]))
            if w not in B:
                rows.append((q, s, w, ad[q] == ad[s], n))
for k, v in sorted(c.items(), key=lambda kv: -kv[1]):
    print(k, v)
for k in sorted(ex, key=lambda k: -c[k])[:8]:
    if k[1] == "added":
        print("\n", k)
        for e in ex[k][:6]:
            print("   ", e)
pl.DataFrame(rows, schema=["q", "s1", "word", "same_addr", "n_acc"], orient="row").write_parquet("/work/lead/fr_acc_dw_added.parquet")

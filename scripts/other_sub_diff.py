"""Teammate submission vs ours, France focus: pairs only in the teammate file, classified by where we put the record
(unassigned / another S1) and by decoy signatures (decoy word, house = S1 house + decoy offset, same-address type swap),
plus agreement on our decisions. Tells whether the teammate carries signal we lack.

python scripts/teammate_diff.py /work/out_fr8b /work/dive/teammate.tsv
"""
import re
import sys
import unicodedata

import polars as pl

ours, theirs = sys.argv[1], sys.argv[2]
DW = {"holding", "participations", "distribution", "international", "groupe", "developpement", "snc", "southside", "eastgate", "northside",
      "midtown", "greater"}
OFFS = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}
LEG = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "inc", "llc", "ltd", "corp", "co", "the", "de", "des", "du", "la", "le", "les", "et", "of", "and"}


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country", "business_name", "addr_n", "house"]) \
        .with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
eid = rec.select("rid", "entity_id")


def load(p):
    g = pl.read_csv(p, separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
    return g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls().filter(pl.col("matched_entity_ids") != "") \
            .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
            .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1")


A, B = load(ours), load(theirs)
ctry = dict(zip(rec["rid"].to_list(), rec["country"].to_list()))
names = rec["business_name"].to_list(); addrs = rec["addr_n"].to_list(); hs = rec["house"].to_list()
nt = [set(re.findall(r"[a-z0-9]+", norm(n))) for n in names]
at = [set(re.findall(r"[a-z0-9]+", norm(a))) if a else set() for a in addrs]
a_q = dict(zip(A["q"].to_list(), A["s1"].to_list()))


def cls(q, s):
    if (nt[q] & DW) - nt[s]:
        return "decoy_word"
    if hs[q] and hs[s] and hs[q].isdigit() and hs[s].isdigit() and int(hs[q]) - int(hs[s]) in OFFS and (at[q] - {hs[q]}) == (at[s] - {hs[s]}):
        return "house_offset"
    nq, ns = nt[q] - LEG, nt[s] - LEG
    if at[q] == at[s] and at[q]:
        if nq == ns:
            return "same_addr_same_name"
        if len(nq - ns) == 1 and len(ns - nq) == 1:
            return "same_addr_swap"
        return "same_addr_other_name"
    if nq == ns:
        return "same_name_addr_diff"
    return "other"


for country in ("US", "India", "France"):
    a = A.filter(pl.col("s1").map_elements(lambda s: ctry[s] == country, return_dtype=pl.Boolean))
    b = B.filter(pl.col("s1").map_elements(lambda s: ctry[s] == country, return_dtype=pl.Boolean))
    only_b = b.join(a, on=["q", "s1"], how="anti")
    where = ["we: unassigned" if a_q.get(q) is None else "we: other S1" for q in only_b["q"].to_list()]
    kinds = [cls(q, s) for q, s in zip(only_b["q"].to_list(), only_b["s1"].to_list())]
    ob = only_b.with_columns(pl.Series("where", where), pl.Series("kind", kinds))
    print(f"\n=== {country}: ours {a.height}, teammate {b.height}, both {a.join(b, on=['q', 's1']).height}, only teammate {only_b.height}")
    print(ob.group_by("where", "kind").len().sort("len", descending=True).head(14).rows())

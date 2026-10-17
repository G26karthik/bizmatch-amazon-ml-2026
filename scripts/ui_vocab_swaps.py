"""US/India test: accepted same-house one-word swaps by kind of new word (generic copy word / vocabulary word / non-word /
near-spelling), with bracket share (copy marker) — does the test's higher decoy density put sibling decoys among
accepted US/India pairs that validation never showed? Validation reference: 37 accepted same-address vocabulary swaps,
100% true.

python scripts/ui_vocab_swaps.py /work/out_fr14
"""
import re
import sys
import unicodedata
from collections import Counter

import polars as pl
from rapidfuzz import fuzz

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(230)
out = sys.argv[1]
LEG = {"inc", "llc", "ltd", "pvt", "private", "limited", "corp", "corporation", "incorporated", "company", "co", "lp", "llp", "pllc",
       "pc", "plc", "the", "of", "and", "s", "a", "&"}
GEN = {"center", "services", "service", "partners", "group", "solutions", "enterprises", "associates", "centre", "industries",
       "international", "global", "holdings", "company"}
DW = {"southside", "eastgate", "northside", "midtown", "greater"}


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


rec = pl.read_parquet("/work/run28/test/records.parquet", columns=["entity_id", "business_name", "country", "src", "addr_n", "house"]).with_row_index("rid") \
        .with_columns(pl.col("rid").cast(pl.UInt32))
nm = rec["business_name"].to_list(); ad = rec["addr_n"].to_list(); hs = rec["house"].to_list()
eid = rec.select("rid", "entity_id")
g = pl.read_csv(f"{out}/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = g.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "") \
       .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
       .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1") \
       .join(rec.select(pl.col("rid").alias("s1"), "country"), on="s1").filter(pl.col("country") != "France")
for c in ("US", "India"):
    s1n = rec.filter((pl.col("country") == c) & (pl.col("src") == 1))["business_name"].to_list()
    voc = Counter(w for n in s1n for w in set(re.findall(r"[a-z0-9]+", norm(n))))
    rows = []
    for q, s in acc.filter(pl.col("country") == c).select("q", "s1").iter_rows():
        if not hs[q] or hs[q] != hs[s]:
            continue
        A = set(re.findall(r"[a-z0-9]+", norm(nm[q]))) - LEG
        B = set(re.findall(r"[a-z0-9]+", norm(nm[s]))) - LEG
        if not (len(A - B) == 1 and len(B - A) == 1 and A & B):
            continue
        wi, wo = next(iter(A - B)), next(iter(B - A))
        near = fuzz.ratio(wi, wo) >= 70 or wi.startswith(wo[:4]) or wo.startswith(wi[:4])
        kind = "generic" if wi in GEN else ("decoyword" if wi in DW else ("near" if near else ("vocab" if voc.get(wi, 0) > 30 else "nonword")))
        br = bool(re.search(r"[\[\(]\s*" + re.escape(wi), norm(nm[q])))
        rows.append((q, s, kind, br, wi, wo))
    x = pl.DataFrame(rows, schema=["q", "s1", "kind", "bracket", "new", "old"], orient="row")
    print(f"\n=== {c} test: accepted same-house one-word swaps (S1s {len(s1n)})")
    print(x.group_by("kind").agg(pl.len().alias("n"), pl.col("bracket").mean().round(4).alias("brk")).sort("n", descending=True))
    v = x.filter(pl.col("kind") == "vocab")
    print("top vocab word pairs:", v.group_by("old", "new").len().sort("len", descending=True).head(15).rows())
    for q, s in v.head(8).select("q", "s1").iter_rows():
        print(f"   {nm[s]!r:44} | {nm[q]!r:44} | {ad[s]}")

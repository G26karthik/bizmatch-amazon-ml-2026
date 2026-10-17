"""Deep dive 22: unassigned one-token S2/S3 records in test (fr19). For each: S1s of the country at the same address
(address token set equal, or Jaccard >= 0.6 with the same house number), the relation of the token to those S1 names
(initials / concatenation handle / fuzzy / none=pseudo-brand) and how many such S1s exist. France vs US.

uv run --no-project --with polars==1.44.2 --with rapidfuzz python scripts/deep/d22_fr_one_token.py [country]
"""
import re
import sys
import unicodedata
from collections import defaultdict

import polars as pl
from rapidfuzz import fuzz

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(200); pl.Config.set_fmt_str_lengths(50)
CTRY = sys.argv[1] if len(sys.argv) > 1 else "France"
LEG = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "de", "des", "du", "la", "le", "les", "et", "the", "and", "of",
       "inc", "llc", "ltd", "limited", "private", "pvt", "corp", "co", "company", "lp", "llp", "pc"}


def toks(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    return re.findall(r"\w+", "".join(ch for ch in s if not unicodedata.combining(ch)))


rec = pl.concat([pl.read_parquet(f"local_data/test_source{s}.parquet") for s in (1, 2, 3)]).filter(pl.col("country") == CTRY)
g = pl.read_csv("output/v28q_fr19/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
assigned = set(g["matched_entity_ids"].fill_null("").str.split(",").explode().to_list())
s1 = rec.filter(pl.col("entity_id").str.starts_with("S1"))
by_house = defaultdict(list)
for e, n, a in s1.select("entity_id", "business_name", "business_address").iter_rows():
    T = set(toks(a)); h = [t for t in T if t.isdigit()]
    for x in (h or ["_"]):
        by_house[x].append((e, n, frozenset(T)))
r = rec.filter(~pl.col("entity_id").str.starts_with("S1") & ~pl.col("entity_id").is_in(list(assigned)))
r = r.filter(~pl.col("business_name").fill_null("").str.strip_chars().str.contains(" ") & (pl.col("business_address").fill_null("").str.strip_chars() != ""))
rows = []
for e, n, a in r.select("entity_id", "business_name", "business_address").iter_rows():
    tok = "".join(toks(n))
    T = set(toks(a)); h = [t for t in T if t.isdigit()]
    near = []
    for x in (h or ["_"]):
        for s, sn, ST in by_house.get(x, []):
            j = len(T & ST) / max(1, len(T | ST))
            if j >= 0.6:
                near.append((s, sn, j))
    rel = "no_s1_at_addr"
    if near:
        best = None
        for s, sn, j in near:
            ws = [w for w in toks(sn) if w not in LEG]
            ini = "".join(w[0] for w in ws)
            conc = "".join(toks(sn))
            if tok and (tok == ini or tok == "".join(w[0] for w in toks(sn))):
                best = "initials"; break
            if fuzz.partial_ratio(tok, conc) >= 85:
                best = best or "handle"
        rel = best or "pseudo"
    rows.append((e, n, a, rel, len(near), tok.isupper() or n.isupper(), len(tok)))
x = pl.DataFrame(rows, schema=["m", "name", "addr", "rel", "n_s1_at_addr", "upper", "len"], orient="row")
print(CTRY, "unassigned one-token records with an address:", x.height)
print(x.group_by("rel", pl.col("n_s1_at_addr").clip(0, 3)).len().sort("rel", "n_s1_at_addr"))
for rel in ("pseudo", "initials", "handle", "no_s1_at_addr"):
    sub = x.filter(pl.col("rel") == rel)
    if sub.height:
        print(rel, "examples:")
        print(sub.sample(min(10, sub.height), seed=2).select("name", "addr", "n_s1_at_addr"))
x.write_parquet(f"local_data/work/one_token_{CTRY}.parquet")

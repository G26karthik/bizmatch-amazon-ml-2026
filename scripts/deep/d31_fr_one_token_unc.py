"""Deep dive 31: French unassigned one-token records (fr22) - best candidate in the uncertain zone (production prob and
test-like recalibrated pn), relation of the token to that S1 (initials / handle / pseudo), address relation, examples.

uv run --no-project --with polars==1.44.2 --with rapidfuzz python scripts/deep/d31_fr_one_token_unc.py
"""
import re
import unicodedata

import polars as pl
from rapidfuzz import fuzz

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(220)
LEG = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "de", "des", "du", "la", "le", "les", "et", "l", "d", "s", "a"}


def toks(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    return re.findall(r"\w+", "".join(ch for ch in s if not unicodedata.combining(ch)))


rec = pl.concat([pl.read_parquet(f"local_data/test_source{s}.parquet") for s in (1, 2, 3)]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
eid = rec["entity_id"].to_list(); nm = rec["business_name"].to_list(); ad = rec["business_address"].to_list(); ct = rec["country"].to_list()
g = pl.read_csv("output/v28q_fr22/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
assigned = set(g["matched_entity_ids"].fill_null("").str.split(",").explode().to_list())
fr1 = [i for i in range(len(eid)) if ct[i] == "France" and not eid[i].startswith("S1") and eid[i] not in assigned
       and nm[i] and " " not in nm[i].strip() and (ad[i] or "").strip()]
print("French unassigned one-token records with an address:", len(fr1))
u = pl.read_parquet("local_data/llm/unc_test.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
r = pl.read_parquet("local_data/llm/resid_test_tl_v34.parquet").select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32), "pn")
x = pl.DataFrame({"q": pl.Series(fr1, dtype=pl.UInt32)}).join(u, on="q", how="left").join(r, on=["q", "s1"], how="left")
rows = []
for q, s, p, pn in x.select("q", "s1", "prob", "pn").iter_rows():
    if s is None:
        rows.append(("no_unc_pair", None, None, None, None)); continue
    tok = "".join(toks(nm[q])); ws = [w for w in toks(nm[s]) if w not in LEG]
    rel = "initials" if ws and tok == "".join(w[0] for w in ws) else ("handle" if len(tok) >= 6 and fuzz.partial_ratio(tok, "".join(toks(nm[s]))) >= 90 else "pseudo")
    A, B = set(toks(ad[q])), set(toks(ad[s]))
    da = sorted(t for t in A if t.isdigit()); db = sorted(t for t in B if t.isdigit())
    arel = "exact" if A == B else ("same_house" if da and da == db else "house_diff")
    rows.append((rel, arel, p, pn, s))
y = pl.DataFrame(rows, schema=["rel", "arel", "p", "pn", "s1"], orient="row")
print(y.group_by("rel", "arel").agg(pl.len(), pl.col("p").mean().round(3), pl.col("pn").mean().round(3)).sort("rel", "arel"))
z = pl.concat([x.select("q"), y], how="horizontal").filter(pl.col("rel") == "pseudo")
for q, s, arel, p in z.sample(min(20, z.height), seed=1).select("q", "s1", "arel", "p").iter_rows():
    print(f"   {nm[q]!r:18.18s} | {ad[q]!s:45.45s} -> {nm[s]!r:32.32s} | {ad[s]!s:45.45s} [{arel} p={p:.2f}]")

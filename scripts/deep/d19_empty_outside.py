"""Deep dive 19: unassigned empty-address US/India test records in fr17: in the uncertain zone (best pair 0.02<p<0.98)
or not; for the ones outside, does a simple token search find a unique S1 whose name tokens (minus legal forms) are
all contained in the record's tokens (subset) or equal? Counts + examples.

uv run --no-project --with polars==1.44.2 python scripts/deep/d19_empty_outside.py
"""
import re
import unicodedata
from collections import defaultdict

import polars as pl

LEG = {"inc", "llc", "ltd", "limited", "private", "pvt", "corp", "corporation", "co", "company", "incorporated", "lp", "llp",
       "pc", "plc", "the", "and", "of", "l", "c", "p", "d", "s", "a"}


def toks(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    return re.findall(r"\w+", "".join(ch for ch in s if not unicodedata.combining(ch)))


rec = pl.concat([pl.read_parquet(f"local_data/test_source{s}.parquet") for s in (1, 2, 3)]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
g = pl.read_csv("output/v28q_fr17/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
assigned = set(g["matched_entity_ids"].fill_null("").str.split(",").explode().to_list())
unc_q = set(pl.read_parquet("local_data/llm/unc_test.parquet")["q"].cast(pl.UInt32).to_list())
s1 = rec.filter(pl.col("entity_id").str.starts_with("S1") & (pl.col("country") != "France"))
by_tok = defaultdict(set); s1tok = {}
for rid, n, c in s1.select("rid", "business_name", "country").iter_rows():
    t = frozenset(set(toks(n)) - LEG); s1tok[rid] = (t, c)
    for w in t:
        by_tok[(c, w)].add(rid)
e = rec.filter(~pl.col("entity_id").str.starts_with("S1") & (pl.col("country") != "France") & (pl.col("business_address").fill_null("").str.strip_chars() == ""))
e = e.filter(~pl.col("entity_id").is_in(list(assigned)))
print("unassigned empty-address US/India records:", e.height, " in uncertain zone:", e.filter(pl.col("rid").is_in(list(unc_q))).height)
out = e.filter(~pl.col("rid").is_in(list(unc_q)))
stats = defaultdict(int); ex = defaultdict(list)
nm = dict(zip(rec["rid"].to_list(), rec["business_name"].to_list()))
for rid, n, c in out.select("rid", "business_name", "country").iter_rows():
    T = set(toks(n)) - LEG
    if not T:
        stats["no_tokens"] += 1; continue
    rare = min(T, key=lambda w: len(by_tok.get((c, w), ())))
    cands = [s for s in by_tok.get((c, rare), ()) if s1tok[s][0] <= T]
    eq = [s for s in cands if s1tok[s][0] == T]
    sub = [s for s in cands if s1tok[s][0] < T]
    k = "eq1" if len(eq) == 1 else ("eq_multi" if eq else ("sub1" if len(sub) == 1 else ("sub_multi" if sub else "none")))
    stats[k] += 1
    if len(ex[k]) < 8:
        ex[k].append((n, [nm[s] for s in (eq or sub)[:3]]))
print("outside the zone:", out.height, dict(stats))
for k, v in ex.items():
    print(k)
    for x in v:
        print("   ", x)

# ---- why are eq1/sub1 records not decided? cap status of the unique S1 and whether the pair is a scored candidate
cnt = defaultdict(int)
eid = rec["entity_id"].to_list()
rid_of = {x: i for i, x in enumerate(eid)}
for s, m in zip(g["source1_entity_id"].to_list(), g["matched_entity_ids"].fill_null("").to_list()):
    for q in (m.split(",") if m else []):
        cnt[(rid_of[s], q[:2])] += 1
cp = pl.read_csv("output/v28q_fr17/candidate_pairs.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "candidate_entity_ids": pl.String})
cand = {}
for s, m in zip(cp[cp.columns[0]].to_list(), cp[cp.columns[1]].fill_null("").to_list()):
    for q in (m.split(",") if m else []):
        cand.setdefault(q, set()).add(s)
res = defaultdict(int)
for rid, n, c in out.select("rid", "business_name", "country").iter_rows():
    T = set(toks(n)) - LEG
    if not T:
        continue
    rare = min(T, key=lambda w: len(by_tok.get((c, w), ())))
    eq = [s for s in by_tok.get((c, rare), ()) if s1tok[s][0] == T]
    if len(eq) != 1:
        continue
    s = eq[0]; src = eid[rid][:2]
    full = cnt[(s, src)] >= (5 if src == "S2" else 6)
    incand = eid[s] in cand.get(eid[rid], set())
    othercand = len(cand.get(eid[rid], set()) - {eid[s]})
    res[(("full" if full else "room"), ("scored" if incand else "not_scored"), ("has_other_cands" if othercand else "no_other"))] += 1
print("eq1 records by cap / candidate status:", dict(res))

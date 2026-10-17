"""fr18 = fr17 + unique exact-token matches for never-scored US/India empty-address records (test has no hidden-S1
copies, so an empty-address record is a copy of a visible S1 with ~97.7%): the record's name tokens minus legal forms
equal those of exactly one S1 of its country, the pair is not in the record's candidate list (retrieval miss), the
S1 holding the record's rarest token count <= 100 (train truth, all S1 visible: precision 0.983-0.993), cap room.

uv run --no-project --with polars==1.44.2 python scripts/fr18_build.py
"""
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import polars as pl

LEG = {"inc", "llc", "ltd", "limited", "private", "pvt", "corp", "corporation", "co", "company", "incorporated", "lp", "llp",
       "pc", "plc", "the", "and", "of", "l", "c", "p", "d", "s", "a"}
base, out = Path("output/v28q_fr17"), Path("output/v28q_fr18")


def toks(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    return re.findall(r"\w+", "".join(ch for ch in s if not unicodedata.combining(ch)))


rec = pl.concat([pl.read_parquet(f"local_data/test_source{s}.parquet") for s in (1, 2, 3)])
g = pl.read_csv(base / "matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
order = g["source1_entity_id"].to_list()
members = {s: (set(m.split(",")) if m else set()) for s, m in zip(order, g["matched_entity_ids"].fill_null("").to_list())}
assigned = {e for ms in members.values() for e in ms}
c = pl.read_csv(base / "candidate_pairs.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "candidate_entity_ids": pl.String})
cand = defaultdict(set)
for s, m in zip(c["source1_entity_id"].to_list(), c["candidate_entity_ids"].fill_null("").to_list()):
    for q in (m.split(",") if m else []):
        cand[q].add(s)
s1 = rec.filter(pl.col("entity_id").str.starts_with("S1") & (pl.col("country") != "France"))
by_tok = defaultdict(set); s1tok = {}
for e, n, ct in s1.select("entity_id", "business_name", "country").iter_rows():
    t = frozenset(set(toks(n)) - LEG); s1tok[e] = t
    for w in t:
        by_tok[(ct, w)].add(e)
emp = rec.filter(~pl.col("entity_id").str.starts_with("S1") & (pl.col("country") != "France") & (pl.col("business_address").fill_null("").str.strip_chars() == ""))
adds, skipped = [], defaultdict(int)
for m, n, ct in emp.select("entity_id", "business_name", "country").iter_rows():
    if m in assigned:
        continue
    T = set(toks(n)) - LEG
    if not T:
        continue
    rare = min(T, key=lambda w: len(by_tok.get((ct, w), ())))
    crowd = len(by_tok.get((ct, rare), ()))
    eq = [s for s in by_tok.get((ct, rare), ()) if s1tok[s] == T]
    if len(eq) != 1:
        continue
    s = eq[0]
    if s in cand.get(m, ()):
        skipped["scored"] += 1; continue
    if crowd > 100:
        skipped["crowd>100"] += 1; continue
    if sum(1 for x in members[s] if x[:2] == m[:2]) >= (5 if m[:2] == "S2" else 6):
        skipped["cap"] += 1; continue
    adds.append((m, s, ct, crowd))
added = 0
for m, s, ct, crowd in adds:
    if m in assigned:
        continue
    members[s].add(m); assigned.add(m); added += 1
print(f"added {added} pairs (US {sum(1 for a in adds if a[2] == 'US')}, India {sum(1 for a in adds if a[2] == 'India')}); skipped {dict(skipped)}")
out.mkdir(parents=True, exist_ok=True)
with open(out / "matching_results.tsv", "w", encoding="utf-8", newline="\n") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for s in order:
        f.write(f"{s}\t{','.join(sorted(members[s]))}\n")
with open(out / "candidate_pairs.tsv", "w", encoding="utf-8", newline="\n") as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")
    for s, m in zip(c["source1_entity_id"].to_list(), c["candidate_entity_ids"].fill_null("").to_list()):
        have = m.split(",") if m else []
        extra = sorted(members[s] - set(have))
        f.write(f"{s}\t{','.join(have + extra)}\n")
pl.DataFrame(adds, schema=["m", "s1", "country", "crowd"], orient="row").write_parquet(out / "eq1_adds.parquet")
print("wrote", out)

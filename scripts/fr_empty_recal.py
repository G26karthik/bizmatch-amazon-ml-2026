"""France empty-address adds from the test-like recalibrator (resid_model.py --testlike --extra orphan --test) and the
test-like uncapped scorer (France tier 0 = unique exact name key). Rationale (label-free): empty-address records per S1
are the same in France (0.166) as in US/India, so French decoys carry no empty addresses and an empty-address French
record is a copy of a visible S1 (~97.7%). Only ADDS of currently unassigned empty-address records; pairs whose name
differs from the S1 by a one-word swap to a French S1 vocabulary word are skipped (decoy semantics); French accepted
pairs are kept (p 0.999) and the exact expected-F0.5 choice (production strictness 1.5, caps) decides the adds.

uv run --no-project --with-requirements requirements.txt python scripts/fr_empty_recal.py <base> <out>
"""
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import polars as pl

sys.path.insert(0, "src")
from ber.stage2 import _chosen  # noqa: E402

SP = Path("local_data/work")
base, out = Path(sys.argv[1]), Path(sys.argv[2])
LEG = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "de", "des", "du", "la", "le", "les", "et", "s", "a", "d", "l", "the"}


def toks(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    return re.findall(r"\w+", "".join(ch for ch in s if not unicodedata.combining(ch)))


rec = pl.concat([pl.read_parquet(f"local_data/test_source{s}.parquet") for s in (1, 2, 3)]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
eid = rec["entity_id"].to_list(); nm = rec["business_name"].to_list()
rid_of = {e: i for i, e in enumerate(eid)}
fr = rec.filter(pl.col("country") == "France")
voc = Counter(w for n in fr.filter(pl.col("entity_id").str.starts_with("S1"))["business_name"].to_list() for w in set(toks(n)))
empty = set(fr.filter(pl.col("business_address").fill_null("").str.strip_chars() == "")["rid"].to_list())
fr_s1 = set(fr.filter(pl.col("entity_id").str.starts_with("S1"))["rid"].to_list())
g = pl.read_csv(base / "matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
order = g["source1_entity_id"].to_list()
members = {s: (set(m.split(",")) if m else set()) for s, m in zip(order, g["matched_entity_ids"].fill_null("").to_list())}
assigned = {e for ms in members.values() for e in ms}
acc = [(rid_of[e], rid_of[s]) for s, ms in members.items() if rid_of[s] in fr_s1 for e in ms]
cert = pl.DataFrame(acc, schema={"q": pl.UInt32, "s1": pl.UInt32}, orient="row").with_columns(pl.lit(0.999).alias("prob"))


def vocab_swap(q, s):
    A, B = set(toks(nm[q])) - LEG, set(toks(nm[s])) - LEG
    return len(A - B) == 1 and len(B - A) == 1 and voc.get(next(iter(A - B)), 0) > 30


r = pl.read_parquet("local_data/llm/resid_test_tl_orphan.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
r = r.filter(pl.col("s1").is_in(list(fr_s1)) & pl.col("q").is_in(list(empty)))
r = r.filter(~pl.col("q").is_in([rid_of[e] for e in assigned]))
r = r.filter(pl.Series([not vocab_swap(q, s) for q, s in zip(r["q"].to_list(), r["s1"].to_list())]))
print("French unassigned empty-address uncertain pairs (no vocab swap):", r.height, " mean prob", round(r["prob"].mean(), 3), "-> pn", round(r["pn"].mean(), 3))
best = pl.concat([cert, r.select("q", "s1", pl.col("pn").cast(pl.Float64).alias("prob"))])
ch = _chosen(best, {"mode": "efopt", "miss": 0.0, "gamma": 1.5}).select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
adds = ch.join(r.select("q", "s1", "prob", "pn"), on=["q", "s1"], how="inner")
lost = cert.join(ch, on=["q", "s1"], how="anti").height
print(f"recalibrator adds {adds.height} (mean pn {adds['pn'].mean() if adds.height else 0:.3f}); accepted pairs the rule would drop (kept anyway): {lost}")
n1 = 0
for q, s in adds.select("q", "s1").iter_rows():
    e, s_ = eid[q], eid[s]
    if e in assigned or sum(1 for x in members[s_] if x[:2] == e[:2]) >= (5 if e[:2] == "S2" else 6):
        continue
    members[s_].add(e); assigned.add(e); n1 += 1
u = pl.read_parquet(SP / "uncapped_adds.parquet").filter(pl.col("country") == "France")
n2 = 0
for e, s in u.select("m", "s1").iter_rows():
    if e in assigned or s not in members or sum(1 for x in members[s] if x[:2] == e[:2]) >= (5 if e[:2] == "S2" else 6):
        continue
    members[s].add(e); assigned.add(e); n2 += 1
print(f"added: recalibrator {n1}, uncapped tier-0 {n2}")
for q, s in adds.sample(min(15, adds.height), seed=1).select("q", "s1").iter_rows():
    print(f"   {nm[q]!r:45s} -> {nm[s]!r}")
out.mkdir(parents=True, exist_ok=True)
with open(out / "matching_results.tsv", "w", encoding="utf-8", newline="\n") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for s in order:
        f.write(f"{s}\t{','.join(sorted(members[s]))}\n")
c = pl.read_csv(base / "candidate_pairs.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "candidate_entity_ids": pl.String})
with open(out / "candidate_pairs.tsv", "w", encoding="utf-8", newline="\n") as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")
    for s, m in zip(c["source1_entity_id"].to_list(), c["candidate_entity_ids"].fill_null("").to_list()):
        have = m.split(",") if m else []
        extra = sorted(members[s] - set(have))
        f.write(f"{s}\t{','.join(have + extra)}\n")
print("wrote", out)

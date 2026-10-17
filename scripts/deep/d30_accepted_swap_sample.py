"""Deep dive 30: sample of fr22 accepted US/India pairs at the S1's exact address (all house numbers equal, token Jaccard
>= 0.6) whose names differ by exactly one replaced non-legal token (not a typo: edit distance > 2), with the production
probability when the pair is in the uncertain zone; train truth rate of the same class for comparison.

uv run --no-project --with polars==1.44.2 --with rapidfuzz python scripts/deep/d30_accepted_swap_sample.py
"""
import re
import unicodedata

import polars as pl
from rapidfuzz.distance import Levenshtein

LEG = {"inc", "llc", "ltd", "limited", "private", "pvt", "corp", "corporation", "co", "company", "incorporated", "lp", "llp",
       "pc", "plc", "the", "and", "of", "l", "c", "p", "d", "s", "a"}


def toks(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    return re.findall(r"\w+", "".join(ch for ch in s if not unicodedata.combining(ch)))


def same_addr(a, b):
    A, B = set(toks(a)), set(toks(b))
    da = sorted(t.lstrip("0") for t in A if t.isdigit()); db = sorted(t.lstrip("0") for t in B if t.isdigit())
    return bool(da) and da == db and len(A & B) / max(1, len(A | B)) >= 0.6


def swap(n1, n2):
    A, B = set(toks(n1)) - LEG, set(toks(n2)) - LEG
    if len(A - B) == 1 and len(B - A) == 1:
        a, b = next(iter(A - B)), next(iter(B - A))
        return Levenshtein.distance(a, b) > 2 and min(len(a), len(b)) >= 3
    return False


def cls(split, pairs, nm, ad):
    out = []
    for s, m in pairs:
        if swap(nm[m], nm[s]) and same_addr(ad[m], ad[s]):
            out.append((s, m))
    return out


tp = pl.read_parquet("local_data/train_pairs.parquet")
tr = pl.concat([pl.read_parquet(f"local_data/train_source{s}.parquet") for s in (1, 2, 3)])
nm = dict(zip(tr["entity_id"], tr["business_name"])); ad = dict(zip(tr["entity_id"], tr["business_address"])); ct = dict(zip(tr["entity_id"], tr["country"]))
s1 = tr.filter(pl.col("entity_id").str.starts_with("S1"))
# train: every S2/S3 record at the exact address of an S1 with a one-token swap (copies AND non-copies), sampled S1s
samp = set(s1.sample(200_000, seed=3)["entity_id"].to_list())
by_addr = {}
for e, a in s1.select("entity_id", "business_address").iter_rows():
    if e in samp:
        by_addr.setdefault(" ".join(sorted(toks(a))), []).append(e)
truth = dict(zip(tp["m"].to_list(), tp["s1"].to_list()))
n_copy = n_non = 0
for e, n, a in tr.filter(~pl.col("entity_id").str.starts_with("S1")).select("entity_id", "business_name", "business_address").iter_rows():
    for s in by_addr.get(" ".join(sorted(toks(a))), ()):
        if swap(n, nm[s]):
            if truth.get(e) == s:
                n_copy += 1
            else:
                n_non += 1
print(f"TRAIN (200k S1 sample, exact address key): one-token-swap records: copies {n_copy}, non-copies {n_non}, P(copy) {n_copy / max(1, n_copy + n_non):.3f}; copies per 1k S1 {n_copy / 200:.2f}")
te = pl.concat([pl.read_parquet(f"local_data/test_source{s}.parquet") for s in (1, 2, 3)])
nm2 = dict(zip(te["entity_id"], te["business_name"])); ad2 = dict(zip(te["entity_id"], te["business_address"])); ct2 = dict(zip(te["entity_id"], te["country"]))
g = pl.read_csv("output/v28q_fr25/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
a = g.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
a = a.filter(pl.col("source1_entity_id").is_in([e for e, c in ct2.items() if c != "France" and e.startswith("S1")]))
hits = [(s, m) for s, m in zip(a["source1_entity_id"].to_list(), a["matched_entity_ids"].to_list()) if swap(nm2[m], nm2[s]) and same_addr(ad2[m], ad2[s])]
n1 = sum(1 for e, c in ct2.items() if c != "France" and e.startswith("S1"))
print(f"TEST accepted US/India one-token-swap pairs at the exact address: {len(hits)} ({len(hits) / n1 * 1000:.2f} per 1k S1)")
import random
random.seed(1)
for s, m in random.sample(hits, min(25, len(hits))):
    print(f"   {nm2[m]!r:45.45s} -> {nm2[s]!r:45.45s} | {ad2[s]!s:40.40s}")

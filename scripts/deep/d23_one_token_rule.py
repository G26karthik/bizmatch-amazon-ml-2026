"""Deep dive 23: one-token records (pseudo-brand / initials / handle) resolved by a STRICT address key: house number +
street tokens after dropping tokens present in > 1% of the country's S1 addresses (city/region/generic words).
Rule: record unassigned, S1s at the same strict key = candidates; initials or handle relation to exactly one candidate
-> that S1; pseudo-brand (no relation) with exactly one candidate -> that S1. Train truth (US/India, all S1 visible =
test-like) gives the precision of each (relation, n_candidates) class; test (fr19) gives counts for every country.

uv run --no-project --with polars==1.44.2 --with rapidfuzz python scripts/deep/d23_one_token_rule.py
"""
import re
import unicodedata
from collections import Counter, defaultdict

import polars as pl
from rapidfuzz import fuzz

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(200)
LEG = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "de", "des", "du", "la", "le", "les", "et", "the", "and", "of",
       "inc", "llc", "ltd", "limited", "private", "pvt", "corp", "co", "company", "lp", "llp", "pc", "l", "d", "s", "a"}
OUT = "local_data/work/one_token_rule_{}.parquet"


def toks(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    return re.findall(r"\w+", "".join(ch for ch in s if not unicodedata.combining(ch)))


def relation(tok, sname):
    ws = [w for w in toks(sname) if w not in LEG]
    if not ws:
        return "pseudo"
    if tok in ("".join(w[0] for w in ws), "".join(w[0] for w in toks(sname))):
        return "initials"
    if len(tok) >= 5 and fuzz.partial_ratio(tok, "".join(toks(sname))) >= 90:
        return "handle"
    return "pseudo"


def run(split, assigned=None, truth=None):
    rec = pl.concat([pl.read_parquet(f"local_data/{split}_source{s}.parquet") for s in (1, 2, 3)])
    res = []
    for ctry in rec["country"].unique().to_list():
        r = rec.filter(pl.col("country") == ctry)
        s1 = r.filter(pl.col("entity_id").str.starts_with("S1"))
        df = Counter(t for a in s1["business_address"].to_list() for t in set(toks(a)))
        common = {t for t, c in df.items() if c > 0.01 * s1.height}
        akey = lambda a: frozenset(t for t in toks(a) if t not in common)  # noqa: E731
        by = defaultdict(list)
        full = {}
        for e, n, a in s1.select("entity_id", "business_name", "business_address").iter_rows():
            k = akey(a)
            full[e] = set(toks(a))
            if any(t.isdigit() for t in k) and len(k) >= 2:
                by[k].append((e, n))

        def same_addr(a, e):   # all digit tokens equal (zero-stripped) and full token Jaccard >= 0.6
            A, B = set(toks(a)), full[e]
            da = sorted(t.lstrip("0") for t in A if t.isdigit()); db = sorted(t.lstrip("0") for t in B if t.isdigit())
            return da == db and len(A & B) / max(1, len(A | B)) >= 0.6
        q = r.filter(~pl.col("entity_id").str.starts_with("S1") & ~pl.col("business_name").fill_null("").str.strip_chars().str.contains(" "))
        if assigned is not None:
            q = q.filter(~pl.col("entity_id").is_in(assigned))
        for m, n, a in q.select("entity_id", "business_name", "business_address").iter_rows():
            k = akey(a)
            cands = [(s_, sn) for s_, sn in by.get(k, []) if same_addr(a, s_)]
            if not cands:
                continue
            tok = "".join(toks(n))
            rels = [(s, relation(tok, sn)) for s, sn in cands]
            named = [s for s, rl in rels if rl != "pseudo"]
            if len(named) == 1:
                kind, s = next(rl for s_, rl in rels if s_ == named[0]), named[0]
            elif not named and len(cands) == 1:
                kind, s = "pseudo", cands[0][0]
            else:
                kind, s = ("ambiguous_named" if named else "pseudo_multi"), None
            res.append((ctry, m, s, kind, len(cands), (truth.get(m) == s) if (truth is not None and s) else None, (truth.get(m) is None) if truth is not None else None))
    return pl.DataFrame(res, schema=["country", "m", "s1", "kind", "n_cand", "right", "noise"], orient="row")


tp = pl.read_parquet("local_data/train_pairs.parquet")
tr = run("train", truth=dict(zip(tp["m"].to_list(), tp["s1"].to_list())))
print("TRAIN (all one-token records, test-like truth):")
print(tr.group_by("country", "kind", pl.col("n_cand").clip(1, 3)).agg(pl.len(), pl.col("right").mean().round(4).alias("P_right"),
      pl.col("noise").mean().round(4).alias("P_noise")).sort("country", "kind", "n_cand"))
g = pl.read_csv("output/v28q_fr19/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
assigned = g["matched_entity_ids"].fill_null("").str.split(",").explode().implode()
te = run("test", assigned=assigned)
print("TEST unassigned one-token records resolved by the rule:")
print(te.group_by("country", "kind", pl.col("n_cand").clip(1, 3)).len().sort("country", "kind", "n_cand"))
te.write_parquet(OUT.format("test")); tr.write_parquet(OUT.format("train"))

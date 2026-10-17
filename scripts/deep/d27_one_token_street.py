"""Deep dive 27: unassigned one-token records (initials / handle) whose address matches no S1 exactly: S1s on the same
street (address token Jaccard >= 0.6 after removing digits) with an initials/handle relation; unique -> candidate.
Train truth (all visible) precision; test counts per country (fr20).

uv run --no-project --with polars==1.44.2 --with rapidfuzz python scripts/deep/d27_one_token_street.py
"""
import re
import unicodedata
from collections import defaultdict

import polars as pl
from rapidfuzz import fuzz

pl.Config.set_tbl_rows(-1)
LEG = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "de", "des", "du", "la", "le", "les", "et", "the", "and", "of",
       "inc", "llc", "ltd", "limited", "private", "pvt", "corp", "co", "company", "lp", "llp", "pc", "l", "d", "s", "a"}
OUT = "local_data/work/one_token_street_{}.parquet"


def toks(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    return re.findall(r"\w+", "".join(ch for ch in s if not unicodedata.combining(ch)))


def rel(tok, sname):
    ws = [w for w in toks(sname) if w not in LEG]
    if ws and tok in ("".join(w[0] for w in ws), "".join(w[0] for w in toks(sname))) and len(tok) >= 2:
        return "initials"
    if len(tok) >= 6 and fuzz.partial_ratio(tok, "".join(toks(sname))) >= 92:
        return "handle"
    return None


def run(split, assigned=None, truth=None):
    rec = pl.concat([pl.read_parquet(f"local_data/{split}_source{s}.parquet") for s in (1, 2, 3)])
    res = []
    for ctry in rec["country"].unique().to_list():
        r = rec.filter(pl.col("country") == ctry)
        s1 = r.filter(pl.col("entity_id").str.starts_with("S1"))
        by_tok = defaultdict(list)
        S = {}
        for e, n, a in s1.select("entity_id", "business_name", "business_address").iter_rows():
            T = frozenset(t for t in toks(a) if not t.isdigit()); S[e] = (n, T, set(toks(a)))
            for w in T:
                by_tok[w].append(e)
        df = {w: len(v) for w, v in by_tok.items()}
        q = r.filter(~pl.col("entity_id").str.starts_with("S1") & ~pl.col("business_name").fill_null("").str.strip_chars().str.contains(" ")
                     & (pl.col("business_address").fill_null("").str.strip_chars() != ""))
        if assigned is not None:
            q = q.filter(~pl.col("entity_id").is_in(assigned))
        elif split == "train":
            q = q.sample(fraction=0.5, seed=2)
        for m, n, a in q.select("entity_id", "business_name", "business_address").iter_rows():
            T = frozenset(t for t in toks(a) if not t.isdigit())
            if not T:
                continue
            rare = sorted(T, key=lambda w: df.get(w, 0))[:2]
            pool = set(by_tok.get(rare[0], [])) | (set(by_tok.get(rare[1], [])) if len(rare) > 1 else set())
            if len(pool) > 3000:
                continue
            tok = "".join(toks(n))
            hits = []
            for s in pool:
                sn, ST, SA = S[s]
                if len(T & ST) / max(1, len(T | ST)) < 0.6:
                    continue
                r_ = rel(tok, sn)
                if r_:
                    hits.append((s, r_, SA == set(toks(a))))
            if len(hits) != 1:
                continue
            s, r_, exact = hits[0]
            res.append((ctry, m, s, r_, exact, (truth.get(m) == s) if truth is not None else None))
    return pl.DataFrame(res, schema=["country", "m", "s1", "kind", "exact_addr", "right"], orient="row")


tp = pl.read_parquet("local_data/train_pairs.parquet")
tr = run("train", truth=dict(zip(tp["m"].to_list(), tp["s1"].to_list())))
print("TRAIN precision (unique initials/handle S1 on the same street):")
print(tr.group_by("country", "kind", "exact_addr").agg(pl.len(), pl.col("right").mean().round(4).alias("P_right")).sort("country", "kind", "exact_addr"))
g = pl.read_csv("output/v28q_fr20/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
te = run("test", assigned=g["matched_entity_ids"].fill_null("").str.split(",").explode().implode())
print("TEST unassigned:")
print(te.group_by("country", "kind", "exact_addr").len().sort("country", "kind", "exact_addr"))
te.write_parquet(OUT.format("test"))

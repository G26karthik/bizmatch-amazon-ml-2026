"""Deep dive 32: one-token records (pseudo-brand / initials / handle) matched to the ONLY S1 at a format-tolerant
equivalent address: house-number tokens equal (zero-stripped; 'b'/'bis'/'ter' ignored), street tokens (no digits, not
in >1% of the country's S1 addresses, len >= 3) Jaccard >= 0.5, full-token Jaccard >= 0.35. Uniqueness over all S1s
of the country sharing a rare street token. Train truth (all S1 visible = test-like) precision; test unassigned counts
(fr22), split by whether the model scored the pair (uncertain-zone best / rejected / never a candidate).

uv run --no-project --with polars==1.44.2 --with rapidfuzz python scripts/deep/d32_one_token_fuzzyaddr.py
"""
import re
import unicodedata
from collections import Counter, defaultdict

import polars as pl
from rapidfuzz import fuzz

pl.Config.set_tbl_rows(-1)
LEG = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "de", "des", "du", "la", "le", "les", "et", "the", "and", "of",
       "inc", "llc", "ltd", "limited", "private", "pvt", "corp", "co", "company", "lp", "llp", "pc", "l", "d", "s", "a"}
OUT = "local_data/work/one_token_fuzzy_{}.parquet"


def toks(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    return re.findall(r"\w+", "".join(ch for ch in s if not unicodedata.combining(ch)))


def rel(tok, sname):
    ws = [w for w in toks(sname) if w not in LEG]
    if ws and len(tok) >= 2 and tok in ("".join(w[0] for w in ws), "".join(w[0] for w in toks(sname))):
        return "initials"
    if len(tok) >= 6 and fuzz.partial_ratio(tok, "".join(toks(sname))) >= 90:
        return "handle"
    return "pseudo"


def run(split, keep=None, truth=None):
    rec = pl.concat([pl.read_parquet(f"local_data/{split}_source{s}.parquet") for s in (1, 2, 3)])
    res = []
    for ctry in rec["country"].unique().to_list():
        r = rec.filter(pl.col("country") == ctry)
        s1 = r.filter(pl.col("entity_id").str.starts_with("S1"))
        df = Counter(t for a in s1["business_address"].to_list() for t in set(toks(a)))
        common = {t for t, c in df.items() if c > 0.01 * s1.height}

        def parts(a):
            T = toks(a)
            h = sorted(t.lstrip("0") for t in T if t.isdigit())
            st = {t for t in T if not t.isdigit() and t not in common and len(t) >= 3 and t not in ("bis", "ter")}
            return h, st, set(T)
        info, by = {}, defaultdict(list)
        for e, n, a in s1.select("entity_id", "business_name", "business_address").iter_rows():
            h, st, T = parts(a); info[e] = (n, h, st, T)
            if h:
                for w in st:
                    by[(tuple(h), w)].append(e)
        q = r.filter(~pl.col("entity_id").str.starts_with("S1") & ~pl.col("business_name").fill_null("").str.strip_chars().str.contains(" ")
                     & (pl.col("business_address").fill_null("").str.strip_chars() != ""))
        q = q.filter(pl.col("entity_id").is_in(keep)) if keep is not None else q.sample(fraction=0.5, seed=5)
        for m, n, a in q.select("entity_id", "business_name", "business_address").iter_rows():
            h, st, T = parts(a)
            if not h or not st:
                continue
            pool = set()
            for w in st:
                pool.update(by.get((tuple(h), w), ()))
            cands = []
            for s in pool:
                sn, sh, sst, sT = info[s]
                if sh != h or not sst:
                    continue
                if len(st & sst) / len(st | sst) >= 0.5 and len(T & sT) / len(T | sT) >= 0.35:
                    cands.append(s)
            if len(cands) != 1:
                continue
            s = cands[0]
            res.append((ctry, m, s, rel("".join(toks(n)), info[s][0]), (truth.get(m) == s) if truth is not None else None))
    return pl.DataFrame(res, schema=["country", "m", "s1", "kind", "right"], orient="row")


tp = pl.read_parquet("local_data/train_pairs.parquet")
tr = run("train", truth=dict(zip(tp["m"].to_list(), tp["s1"].to_list())))
print("TRAIN (all visible) precision, one-token record -> only S1 at a format-tolerant equal address:")
print(tr.group_by("country", "kind").agg(pl.len(), pl.col("right").mean().round(4).alias("P_right")).sort("country", "kind"))
g = pl.read_csv("output/v28q_fr22/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
assigned = g["matched_entity_ids"].fill_null("").str.split(",").explode()
allq = pl.concat([pl.read_parquet(f"local_data/test_source{s}.parquet", columns=["entity_id"]) for s in (2, 3)])
te = run("test", keep=allq.filter(~pl.col("entity_id").is_in(assigned.implode()))["entity_id"].implode())
print("TEST unassigned:")
print(te.group_by("country", "kind").len().sort("country", "kind"))
te.write_parquet(OUT.format("test"))

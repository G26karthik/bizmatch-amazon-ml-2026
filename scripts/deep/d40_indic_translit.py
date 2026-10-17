"""Deep dive 40: Indic-script records. Learn a word-level Indic -> Latin dictionary from train truth pairs (Indic
record vs Latin S1 name, equal token counts, aligned by position; keep mappings with support >= 3 and >= 70%
consistency). Transliterate records, then: empty address -> unique S1 by exact nkey; addressed -> unique S1 by exact
nkey among S1s at a format-tolerant equal address (house numbers equal, token Jaccard >= 0.3 on Latin tokens).
Train precision (dictionary from a disjoint half of the S1s); test unassigned counts (fr27).

uv run --no-project --with polars==1.44.2 python scripts/deep/d40_indic_translit.py
"""
import re
import unicodedata
import zlib
from collections import Counter, defaultdict

import polars as pl

pl.Config.set_tbl_rows(-1)
IND = re.compile(r"[\u0900-\u0DFF]")
LEG = {"inc", "llc", "ltd", "limited", "private", "pvt", "corp", "corporation", "co", "company", "incorporated", "lp", "llp",
       "pc", "plc", "the", "and", "of", "l", "c", "p", "d", "s", "a"}
OUT = "local_data/work/indic_{}.parquet"


def toks(s):
    s = unicodedata.normalize("NFKC", s or "").casefold()
    return re.findall(r"[\w\u0900-\u0DFF]+", s)


def lat(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    return re.findall(r"\w+", "".join(ch for ch in s if not unicodedata.combining(ch)))


tr = pl.concat([pl.read_parquet(f"local_data/train_source{s}.parquet") for s in (1, 2, 3)]).filter(pl.col("country") == "India")
tp = pl.read_parquet("local_data/train_pairs.parquet")
nm = dict(zip(tr["entity_id"], tr["business_name"])); ad = dict(zip(tr["entity_id"], tr["business_address"]))
pairs = [(m, s) for s, m in zip(tp["s1"].to_list(), tp["m"].to_list()) if m in nm and s in nm and IND.search(nm[m] or "")]
half = lambda s: zlib.crc32(s.encode()) % 2  # noqa: E731
cnt = defaultdict(Counter)
for m, s in pairs:
    if half(s) != 0:
        continue
    a, b = toks(nm[m]), lat(nm[s])
    if len(a) == len(b):
        for x, y in zip(a, b):
            cnt[x][y] += 1
D = {}
for x, c in cnt.items():
    y, k = c.most_common(1)[0]
    if k >= 3 and k / sum(c.values()) >= 0.7:
        D[x] = y
print("dictionary entries:", len(D), "from", sum(1 for m, s in pairs if half(s) == 0), "train pairs")


def translit(n):
    out = []
    for t in toks(n):
        if IND.search(t):
            if t not in D:
                return None
            out.append(D[t])
        else:
            out.extend(lat(t))
    return out


def nkey(ts):
    return " ".join(sorted(set(ts) - LEG))


def run(split, keep=None, truth=None, s1_filter=None):
    rec = pl.concat([pl.read_parquet(f"local_data/{split}_source{s}.parquet") for s in (1, 2, 3)]).filter(pl.col("country") == "India")
    s1 = rec.filter(pl.col("entity_id").str.starts_with("S1"))
    by = defaultdict(list); sad = {}
    for e, n, a in s1.select("entity_id", "business_name", "business_address").iter_rows():
        by[nkey(lat(n))].append(e); sad[e] = a
    q = rec.filter(~pl.col("entity_id").str.starts_with("S1") & pl.col("business_name").fill_null("").str.contains(IND.pattern))
    if keep is not None:
        q = q.filter(pl.col("entity_id").is_in(keep))
    res = []
    for m, n, a in q.select("entity_id", "business_name", "business_address").iter_rows():
        if truth is not None and truth.get(m) is not None and half(truth[m]) == 0:
            continue                                   # evaluate only on S1s whose pairs did not build the dictionary
        T = translit(n)
        if not T:
            continue
        cs = by.get(nkey(T), [])
        empty = not (a or "").strip()
        if not empty:
            A = set(lat(a))
            def same(e):
                B = set(lat(sad[e])); da = sorted(t.lstrip("0") for t in A if t.isdigit()); db = sorted(t.lstrip("0") for t in B if t.isdigit())
                return da == db and len(A & B) / max(1, len(A | B)) >= 0.3
            cs = [e for e in cs if same(e)]
        if len(cs) != 1:
            continue
        res.append(("empty" if empty else "addr", m, cs[0], (truth.get(m) == cs[0]) if truth is not None else None))
    return pl.DataFrame(res, schema={"kind": pl.String, "m": pl.String, "s1": pl.String, "right": pl.Boolean}, orient="row")


t = run("train", truth=dict(zip(tp["m"].to_list(), tp["s1"].to_list())))
print("TRAIN precision (held-out half of S1s):"); print(t.group_by("kind").agg(pl.len(), pl.col("right").mean().round(4).alias("P_right")))
g = pl.read_csv("output/v28q_fr27/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
assigned = g["matched_entity_ids"].fill_null("").str.split(",").explode()
allq = pl.concat([pl.read_parquet(f"local_data/test_source{s}.parquet", columns=["entity_id"]) for s in (2, 3)])
te = run("test", keep=allq.filter(~pl.col("entity_id").is_in(assigned.implode()))["entity_id"].implode())
print("TEST unassigned resolved:"); print(te.group_by("kind").len())
te.with_columns(pl.lit("India").alias("country")).write_parquet(OUT.format("test"))

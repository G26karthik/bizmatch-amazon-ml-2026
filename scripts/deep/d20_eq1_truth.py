"""Deep dive 20: train truth (all S1 visible = test-like) precision of simple unique-name rules for empty-address
records: eq1 = exactly one S1 (same country) whose name tokens minus legal forms equal the record's; sub1 = no eq,
exactly one S1 whose tokens are a strict subset. By crowding (S1s holding the rarest token) and extra-token kind.

uv run --no-project --with polars==1.44.2 python scripts/deep/d20_eq1_truth.py
"""
import re
import unicodedata
from collections import defaultdict

import polars as pl

pl.Config.set_tbl_rows(-1)
LEG = {"inc", "llc", "ltd", "limited", "private", "pvt", "corp", "corporation", "co", "company", "incorporated", "lp", "llp",
       "pc", "plc", "the", "and", "of", "l", "c", "p", "d", "s", "a"}
GEN = {"center", "centre", "services", "service", "trading", "enterprises", "group", "partners", "associates", "solutions", "transport",
       "international", "industries", "holdings", "global", "company", "systems", "consulting", "agency", "agencies"}


def toks(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    return re.findall(r"\w+", "".join(ch for ch in s if not unicodedata.combining(ch)))


rec = pl.concat([pl.read_parquet(f"local_data/train_source{s}.parquet") for s in (1, 2, 3)])
tp = pl.read_parquet("local_data/train_pairs.parquet")
truth = dict(zip(tp["m"].to_list(), tp["s1"].to_list()))
s1 = rec.filter(pl.col("entity_id").str.starts_with("S1"))
by_tok = defaultdict(set); s1tok = {}
for e, n, c in s1.select("entity_id", "business_name", "country").iter_rows():
    t = frozenset(set(toks(n)) - LEG); s1tok[e] = t
    for w in t:
        by_tok[(c, w)].add(e)
e = rec.filter(~pl.col("entity_id").str.starts_with("S1") & (pl.col("business_address").fill_null("").str.strip_chars() == ""))
rows = []
for m, n, c in e.select("entity_id", "business_name", "country").iter_rows():
    T = set(toks(n)) - LEG
    if not T:
        continue
    rare = min(T, key=lambda w: len(by_tok.get((c, w), ())))
    crowd = len(by_tok.get((c, rare), ()))
    cands = [s for s in by_tok.get((c, rare), ()) if s1tok[s] <= T]
    eq = [s for s in cands if s1tok[s] == T]
    sub = [s for s in cands if s1tok[s] < T]
    if len(eq) == 1:
        kind, s = "eq1", eq[0]; extra = ""
    elif not eq and len(sub) == 1:
        s = sub[0]; ex = T - s1tok[s]
        kind = "sub1"; extra = "generic" if ex <= GEN else ("dup" if False else "other")
    else:
        continue
    rows.append((c, kind, extra, min(crowd, 999), truth.get(m) == s, truth.get(m) is None))
x = pl.DataFrame(rows, schema=["country", "kind", "extra", "crowd", "right", "noise"], orient="row")
x = x.with_columns(pl.col("crowd").cut([5, 20, 100, 500]).alias("crowd_b"))
print(x.group_by("country", "kind", "extra").agg(pl.len(), pl.col("right").mean().round(4).alias("P_right"), pl.col("noise").mean().round(4).alias("P_noise")).sort("country", "kind", "extra"))
print(x.group_by("kind", "extra", "crowd_b").agg(pl.len(), pl.col("right").mean().round(4).alias("P_right")).sort("kind", "extra", "crowd_b"))

"""Agent FR a4: accepted pairs of fr25 per country by name-op x house x street class (same classifier as a1), per 1k S1,
bracket share of the new word. France vs US/India (US/India accepted ~99% precise) -> French deficits/excesses.

uv run --no-project --with polars==1.44.2 --with rapidfuzz python scripts/agent_fr/a4_country_cmp.py
"""
import re
import unicodedata
from collections import Counter

import polars as pl
from rapidfuzz.distance import Levenshtein

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(250)
SCR = "local_data/work"
LEG = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "inc", "llc", "ltd", "pvt", "private", "limited", "corp", "corporation",
       "incorporated", "company", "co", "lp", "llp", "pllc", "pc", "plc", "the", "de", "des", "du", "la", "le", "les", "et", "of", "and",
       "s", "a", "d", "l", "sarlu", "selarl", "scp", "scm", "gie", "sca", "sel"}
OFFS = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}
BR = re.compile(r"[\[\(\{]([^\]\)\}]*)[\]\)\}]")


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


def tk(s):
    return re.findall(r"[a-z0-9]+", norm(s))


rec = pl.concat([pl.read_parquet(f"local_data/test_source{s}.parquet", columns=["entity_id", "business_name", "business_address", "country"])
                 .with_columns(pl.lit(s, dtype=pl.Int8).alias("src")) for s in (1, 2, 3)])
eid = rec["entity_id"].to_list(); names = rec["business_name"].to_list(); addrs = rec["business_address"].to_list()
ctry = rec["country"].to_list(); srcl = rec["src"].to_list()
rid_of = {e: i for i, e in enumerate(eid)}
voc = {c: Counter() for c in ("France", "US", "India")}
for i in range(len(eid)):
    if srcl[i] == 1:
        voc[ctry[i]].update(set(tk(names[i])) - LEG)
n1 = Counter(ctry[i] for i in range(len(eid)) if srcl[i] == 1)
print(n1, flush=True)
g = pl.read_csv("output/v28q_fr25/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})


def feats(q, s, c):
    rq, rs = set(tk(names[q])), set(tk(names[s]))
    A, B = rq - LEG, rs - LEG
    w = ""
    if not A:
        nr = "only_legal"
    elif A == B:
        nr = "same" if rq == rs else "legal_diff"
    else:
        add, drop = A - B, B - A
        if len(add) == 1 and len(drop) == 1:
            a, d = next(iter(add)), next(iter(drop)); w = a
            nr = "typo1" if Levenshtein.distance(a, d) <= max(1, len(d) // 4) else ("swap_vocab" if voc[c].get(a, 0) > 30 else "swap_nonword")
        elif not drop:
            w = next(iter(add)) if len(add) == 1 else ""
            nr = ("add1" if len(add) == 1 else "addn") + ("_v" if w and voc[c].get(w, 0) > 30 else "")
        elif not add:
            nr = "drop1" if len(drop) == 1 else "dropn"
        elif len(A) == 1:
            nr = "one_token_other"; w = next(iter(A))
        elif not (A & B):
            nr = "disjoint"
        else:
            nr = "multi"
    aq, as_ = tk(addrs[q]), tk(addrs[s])
    if not aq:
        hr = "empty"
    else:
        hq = next((t for t in aq if t.isdigit()), None); hs = next((t for t in as_ if t.isdigit()), None)
        if hq is None or hs is None or len(hq) > 6 or len(hs) > 6:
            hr = "nohouse"
        else:
            d = abs(int(hq) - int(hs)); hr = "same_h" if d == 0 else ("offset" if d in OFFS else "h_other")
    b = set()
    for m in BR.findall(norm(names[q])):
        b |= set(re.findall(r"[a-z0-9]+", m))
    anyb = bool(re.search(r"[\[\(\{]", names[q] or ""))
    return nr, hr, ((w in b) if w else anyb)


rows = []
for s, m in zip(g["source1_entity_id"].to_list(), g["matched_entity_ids"].fill_null("").to_list()):
    si = rid_of[s]; c = ctry[si]
    for q in (m.split(",") if m else []):
        qi = rid_of[q]
        nr, hr, bn = feats(qi, si, c)
        rows.append((c, nr, hr, srcl[qi], bn))
x = pl.DataFrame(rows, schema=["country", "nrel", "hrel", "src", "brk"], orient="row")
x.write_parquet(f"{SCR}/a4_acc.parquet")
ns = pl.DataFrame({"country": list(n1.keys()), "S1s": list(n1.values())})
t = x.group_by("country", "nrel", "hrel").agg(pl.len().alias("n"), pl.col("brk").mean().mul(100).round(2).alias("brk"), (pl.col("src") == 2).mean().round(3).alias("s2")) \
     .join(ns, on="country").with_columns((pl.col("n") / pl.col("S1s") * 1000).round(2).alias("per1k"))
p = t.pivot(on="country", index=["nrel", "hrel"], values=["per1k", "brk", "s2"]).fill_null(0)
p = p.with_columns((pl.col("per1k_France") - (pl.col("per1k_US") + pl.col("per1k_India")) / 2).round(2).alias("FR_minus_avg"))
print(p.sort("per1k_France", descending=True).head(60))
print(p.sort("FR_minus_avg").head(20))
print(t.group_by("country").agg(pl.col("per1k").sum()))

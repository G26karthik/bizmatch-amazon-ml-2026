"""Agent FR a3: French unc_test pairs (best S1, 0.02<p<0.98) by status in fr25 (accepted to this S1 / unassigned / other S1)
x name-op x house x street class, with bracket share of new word, prob mean, cap test (unassigned).

uv run --no-project --with polars==1.44.2 --with rapidfuzz python scripts/agent_fr/a3_unc.py
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
                 .with_columns(pl.lit(s, dtype=pl.Int8).alias("src")) for s in (1, 2, 3)]).with_row_index("rid")
eid = rec["entity_id"].to_list(); names = rec["business_name"].to_list(); addrs = rec["business_address"].to_list()
ctry = rec["country"].to_list(); srcl = rec["src"].to_list()
rid_of = {e: i for i, e in enumerate(eid)}
voc = Counter()
for i in range(len(eid)):
    if srcl[i] == 1 and ctry[i] == "France":
        voc.update(set(tk(names[i])) - LEG)
u = pl.read_parquet("local_data/llm/unc_test.parquet")
u = u.filter(pl.Series([ctry[s] == "France" for s in u["s1"].to_list()]))
g = pl.read_csv("output/v28q_fr25/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
assigned, cnt = {}, Counter()
for s, m in zip(g["source1_entity_id"].to_list(), g["matched_entity_ids"].fill_null("").to_list()):
    si = rid_of[s]
    if ctry[si] != "France":
        continue
    for q in (m.split(",") if m else []):
        qi = rid_of[q]; assigned[qi] = si; cnt[(si, srcl[qi])] += 1


def feats(q, s):
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
            nr = "typo1" if Levenshtein.distance(a, d) <= max(1, len(d) // 4) else ("swap_vocab" if voc.get(a, 0) > 30 else "swap_nonword")
        elif not drop:
            w = next(iter(add)) if len(add) == 1 else ""
            nr = ("add1" if len(add) == 1 else "addn") + ("_v" if w and voc.get(w, 0) > 30 else "")
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
        hr, sr = "empty", "-"
    else:
        hq = next((t for t in aq if t.isdigit()), None); hs = next((t for t in as_ if t.isdigit()), None)
        if hq is None or hs is None or len(hq) > 6 or len(hs) > 6:
            hr = "nohouse"
        else:
            d = abs(int(hq) - int(hs)); hr = "same_h" if d == 0 else ("offset" if d in OFFS else "h_other")
        SA, SB = set(aq) - {hq}, set(as_) - {hs}
        j = len(SA & SB) / max(1, len(SA | SB))
        sr = "st_same" if SA == SB else ("st_near" if j >= 0.5 else "st_far")
    b = set()
    for m in BR.findall(norm(names[q])):
        b |= set(re.findall(r"[a-z0-9]+", m))
    anyb = bool(re.search(r"[\[\(\{]", names[q] or ""))
    return nr, hr, sr, ((w in b) if w else anyb), anyb


rows = []
for q, s, p in zip(u["q"].to_list(), u["s1"].to_list(), u["prob"].to_list()):
    a = assigned.get(q)
    st = "acc" if a == s else ("una" if a is None else "other")
    nr, hr, sr, bn, ba = feats(q, s)
    rows.append((q, s, p, st, nr, hr, sr, srcl[q], cnt[(s, srcl[q])] - (1 if st == "acc" else 0), bn, ba))
x = pl.DataFrame(rows, schema=["q", "s1", "prob", "status", "nrel", "hrel", "srel", "src", "n_other", "brk_new", "brk_any"], orient="row")
x.write_parquet(f"{SCR}/a3_unc.parquet")
slots = pl.read_parquet(f"{SCR}/a1_slots.parquet").filter(pl.col("country") == "France")
full = ((pl.col("src") == 2) & (pl.col("n_other") >= 5)) | ((pl.col("src") == 3) & (pl.col("n_other") >= 6))
low = pl.col("n_other").is_between(2, 3)
den_full = slots.filter(full)["n_slots"].sum(); den_low = slots.filter(low)["n_slots"].sum()
print(x.group_by("status").len())
for st in ("acc", "una"):
    t = x.filter(pl.col("status") == st).group_by("nrel", "hrel", "srel").agg(
        pl.len().alias("n"), pl.col("prob").mean().round(3).alias("p"), pl.col("brk_new").mean().mul(100).round(2).alias("brk%"),
        (pl.col("src") == 2).mean().round(3).alias("s2"), ((full.sum() / den_full) / (low.sum() / den_low)).round(3).alias("cap"))
    print(f"\n=== unc France {st}")
    print(t.filter(pl.col("n") >= 200).sort("n", descending=True).head(45))

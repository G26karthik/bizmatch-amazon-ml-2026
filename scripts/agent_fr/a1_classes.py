"""Agent FR a1: French (and US calibration) S2/S3 records paired with an S1 -> name-op x house x street class, status
(accepted to that S1 / unassigned), label-free markers: bracket share of the new word, S2 share, cap test for
unassigned (rate per S1-source slot at full cap vs n_other 2-3).

uv run --no-project --with polars==1.44.2 --with rapidfuzz python scripts/agent_fr/a1_classes.py
"""
import re
import sys
import unicodedata
from collections import Counter

import polars as pl
from rapidfuzz.distance import Levenshtein

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(250)
OUT = sys.argv[1] if len(sys.argv) > 1 else "output/v28q_fr25"
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


src = []
for s in (1, 2, 3):
    d = pl.read_parquet(f"local_data/test_source{s}.parquet", columns=["entity_id", "business_name", "business_address", "country"])
    src.append(d.with_columns(pl.lit(s, dtype=pl.Int8).alias("src")))
rec = pl.concat(src).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
keep_c = set((sys.argv[2] if len(sys.argv) > 2 else "France").split(","))
eid = rec["entity_id"].to_list(); names = rec["business_name"].to_list(); addrs = rec["business_address"].to_list()
ctry = rec["country"].to_list(); srcl = rec["src"].to_list()
N = len(eid)
rid_of = {e: i for i, e in enumerate(eid)}
print("records", N, flush=True)

nt, raw_nt, brk, house, street = {}, {}, {}, {}, {}
for i in range(N):
    if ctry[i] not in keep_c:
        continue
    r = tk(names[i]); raw_nt[i] = set(r); nt[i] = set(r) - LEG
    b = set()
    for m in BR.findall(norm(names[i])):
        b |= set(re.findall(r"[a-z0-9]+", m))
    brk[i] = b if b else (set() if not re.search(r"[\[\(\{]", names[i] or "") else {"__any__"})
    at = tk(addrs[i])
    h = next((t for t in at if t.isdigit()), None)
    house[i] = int(h) if h is not None and len(h) < 7 else None
    street[i] = " ".join(sorted(set(at) - ({h} if h else set())))
print("tokenised", flush=True)
voc = {c: Counter() for c in keep_c}
for i in range(N):
    if srcl[i] == 1 and ctry[i] in keep_c:
        voc[ctry[i]].update(nt[i])

g = pl.read_csv(f"{OUT}/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = []
for s, m in zip(g["source1_entity_id"].to_list(), g["matched_entity_ids"].fill_null("").to_list()):
    si = rid_of[s]
    if ctry[si] not in keep_c:
        continue
    for q in (m.split(",") if m else []):
        acc.append((rid_of[q], si))
assigned = {q: s for q, s in acc}
cnt = Counter((s, srcl[q]) for q, s in acc)
print("accepted", len(acc), flush=True)


def nrel(q, s, c):
    A, B = nt[q], nt[s]
    if not A:
        return "only_legal", ""
    if A == B:
        return ("same" if raw_nt[q] == raw_nt[s] else "legal_diff"), ""
    add, drop = A - B, B - A
    if len(add) == 1 and len(drop) == 1:
        a, d = next(iter(add)), next(iter(drop))
        if Levenshtein.distance(a, d) <= max(1, len(d) // 4):
            return "typo1", a
        return ("swap_vocab" if voc[c].get(a, 0) > 30 else "swap_nonword"), a
    if not drop:
        a = next(iter(add)) if len(add) == 1 else ""
        return ("add1" if len(add) == 1 else "addn") + ("_v" if a and voc[c].get(a, 0) > 30 else ""), a
    if not add:
        return ("drop1" if len(drop) == 1 else "dropn"), ""
    if len(A) == 1:
        return "one_token_other", next(iter(A))
    if not (A & B):
        return "disjoint", ""
    return "multi", ""


def hrel(q, s):
    if not (addrs[q] or "").strip():
        return "empty"
    hq, hs = house[q], house[s]
    if hq is None or hs is None:
        return "nohouse"
    d = abs(hq - hs)
    return "same_h" if d == 0 else ("offset" if d in OFFS else "h_other")


def srel(q, s):
    if not (addrs[q] or "").strip():
        return "-"
    A, B = set(street[q].split()), set(street[s].split())
    if A == B:
        return "st_same"
    j = len(A & B) / max(1, len(A | B))
    return "st_near" if j >= 0.5 else "st_far"


rows = []
for q, s in acc:
    c = ctry[s]
    nr, w = nrel(q, s, c)
    b = brk[q]
    rows.append((c, "acc", q, s, nr, hrel(q, s), srel(q, s), srcl[q], cnt[(s, srcl[q])] - 1, (w in b) if w else bool(b), bool(b)))
print("acc classified", flush=True)
# unassigned: structural candidates on the same street (house removed), best by name relation
RANK = ["same", "legal_diff", "typo1", "swap_nonword", "swap_vocab", "add1", "add1_v", "drop1", "addn", "addn_v", "dropn", "one_token_other", "multi"]
for c in keep_c:
    by_street = {}
    for i in range(N):
        if ctry[i] == c and srcl[i] == 1 and street[i]:
            by_street.setdefault(street[i], []).append(i)
    for i in range(N):
        if ctry[i] != c or srcl[i] == 1 or i in assigned or not street[i]:
            continue
        if c == "US" and i % 4:
            continue
        best = None
        for s in by_street.get(street[i], []):
            nr, w = nrel(i, s, c)
            if nr not in RANK:
                continue
            hr = hrel(i, s)
            key = (RANK.index(nr), 0 if hr == "same_h" else 1)
            if best is None or key < best[0]:
                best = (key, s, nr, w, hr)
        if best is None:
            continue
        _, s, nr, w, hr = best
        b = brk[i]
        rows.append((c, "una", i, s, nr, hr, "st_same", srcl[i], cnt[(s, srcl[i])], (w in b) if w else bool(b), bool(b)))
print("una classified", flush=True)
x = pl.DataFrame(rows, schema=["country", "status", "q", "s1", "nrel", "hrel", "srel", "src", "n_other", "brk_new", "brk_any"], orient="row")
x.write_parquet(f"{SCR}/a1_pairs.parquet")
n1 = rec.filter(pl.col("src") == 1).group_by("country").len().rename({"len": "S1s"})
# cap-test denominators: S1-source slots by n_other
s1r = [i for i in range(N) if srcl[i] == 1 and ctry[i] in keep_c]
slots = pl.DataFrame([(ctry[s], sr, cnt[(s, sr)]) for s in s1r for sr in (2, 3)], schema=["country", "src", "n_other"], orient="row") \
          .group_by("country", "src", "n_other").agg(pl.len().alias("n_slots"))
slots.write_parquet(f"{SCR}/a1_slots.parquet")
print("done", flush=True)

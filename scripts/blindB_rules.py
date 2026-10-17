"""Blind audit B: hit counts of candidate France rules on the submission (inspection-derived error types).

A. accepted same-house pairs that swap an S1 vocabulary word for another vocabulary word (non-generic), split by
   whether the record also carries an extra op (generic word / legal change) -> sibling decoys the vocab rule missed.
B. accepted initials records ("EP") whose initials fit another S1 at the same address better than the matched S1.
C. unassigned web-handle records (no spaces, *.com / #x / @x) at an S1's exact address whose handle spells that S1.
D. unassigned records at an S1's exact address = S1 name with one word swapped to a generic copy word (fils, cie, ...).
Usage: python scripts/blindB_rules.py <test_dir> <matching_results.tsv> <out_dir>
"""
import re
import sys
import unicodedata
from collections import Counter, defaultdict

import polars as pl
from rapidfuzz import fuzz

TEST, SUB, OUT = sys.argv[1:4]
LEGAL = {"sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "ei", "selarl", "scop"}
STOP = {"de", "du", "des", "la", "le", "les", "l", "d", "et", "en", "a", "au", "aux"}
GENERIC = {"fils", "compagnie", "cie", "associes", "services", "developpement", "groupe", "france", "freres", "frs"}
DECOYW = {"holding", "participations", "distribution", "international", "groupe", "developpement"}
ABBR = {"rue": "r", "avenue": "av", "ave": "av", "boulevard": "bd", "blvd": "bd", "chemin": "ch", "place": "pl",
        "route": "rte", "impasse": "imp", "allee": "all", "quai": "q", "cours": "crs", "passage": "psg", "pass": "psg",
        "residence": "res", "saint": "st", "sainte": "ste"}
REGION = {"hauts", "de", "france", "nord", "pas", "calais", "nouvelle", "aquitaine", "gironde", "pays", "la", "loire",
          "atlantique", "du", "des", "le", "les", "d", "l"}
HN = re.compile(r"(?:^|[\s,#(]|n[o°º]\.?\s?)0*(\d{1,4})(?:\s?(?:bis|ter|[a-d])\b)?(?=[\s,)]|$)", re.I)


def asc(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


def toks(name):
    return [t for t in re.split(r"[^a-z0-9]+", asc(name).replace(".", "")) if t]


def house(a):
    for m in HN.finditer(" " + asc(a)):
        return int(m.group(1))
    return None


def street(a):
    ts = [ABBR.get(t, t) for t in re.split(r"[^a-z0-9]+", asc(a)) if t and not any(c.isdigit() for c in t)]
    return " ".join(sorted({t for t in ts if len(t) > 1 and t not in REGION and t not in ("bis", "ter", "no")}))


def load(k):
    return pl.read_csv(f"{TEST}/test_source{k}.tsv", separator="\t", quote_char=None, infer_schema_length=0).filter(
        pl.col("country") == "France").with_columns(pl.lit(k).alias("src"))


rec = pl.concat([load(k) for k in (1, 2, 3)])
names = dict(zip(rec["entity_id"].to_list(), rec["business_name"].fill_null("").to_list()))
addrs = dict(zip(rec["entity_id"].to_list(), rec["business_address"].fill_null("").to_list()))
s1s = [e for e in names if e.startswith("S1")]
qs = [e for e in names if not e.startswith("S1")]
mr = pl.read_csv(SUB, separator="\t", infer_schema_length=0).with_columns(pl.col("matched_entity_ids").fill_null(""))
assigned, nmatch = {}, Counter()
s1set = set(s1s)
for s, m in mr.iter_rows():
    if s in s1set:
        for q in m.split(","):
            if q:
                assigned[q] = s
                nmatch[s] += 1
df = Counter()
for s in s1s:
    for t in set(toks(names[s])):
        df[t] += 1
VOC = {t for t, c in df.items() if c >= 150 and t not in LEGAL and t not in STOP and t not in GENERIC and len(t) > 2}
key_of = {e: (street(addrs[e]), house(addrs[e])) for e in names}
at_addr = defaultdict(list)
for s in s1s:
    if key_of[s][1] is not None and key_of[s][0]:
        at_addr[key_of[s]].append(s)
rep = open(f"{OUT}/rules_report.txt", "w", encoding="utf-8")
rep.write(f"France S1 {len(s1s)}, records {len(qs)}, vocab size {len(VOC)}\n")

# ---- A: accepted vocab swaps at same house
A = []
for q, s in assigned.items():
    if key_of[q][1] is None or key_of[q] != key_of[s]:
        continue
    tq, ts = set(toks(names[q])) - STOP, set(toks(names[s])) - STOP
    rem, add = (ts - tq) - LEGAL, (tq - ts) - LEGAL
    rv, av = rem & VOC, add & VOC
    if rv and av and not any(fuzz.ratio(a, r) >= 70 or a.startswith(r[:3]) for a in av for r in rv):
        extra = []
        if add & GENERIC:
            extra.append("generic")
        if (tq & LEGAL) != (ts & LEGAL) and (tq & LEGAL) and (ts & LEGAL):
            extra.append("legalswap")
        if names[q] == names[q].lower():
            extra.append("lower")
        A.append((q, s, "+".join(extra) or "none", " ".join(sorted(rv)), " ".join(sorted(av))))
rep.write(f"\nA. accepted same-house vocab swaps: {len(A)} pairs, {len({a[1] for a in A})} S1s\n")
rep.write("   by extra op: " + str(Counter(a[2] for a in A).most_common()) + "\n")
rep.write("   S1s where the swap pair is the only match: "
          f"{sum(1 for a in A if nmatch[a[1]] == 1)}\n")
for a in A[:40]:
    rep.write(f"   [{a[2]}] {names[a[1]]!r} -> {names[a[0]]!r} | {addrs[a[0]]!r}\n")

# ---- B: initials pointing to another S1 at the same address
def initials(name, stop=True):
    t = [x for x in toks(name) if x not in LEGAL and not (stop and x in STOP)]
    return "".join(x[0] for x in t)


B = []
for q, s in assigned.items():
    nq = asc(names[q]).replace(".", "").strip()
    if not re.fullmatch(r"[a-z]{2,4}", nq):
        continue
    own = {initials(names[s]), initials(names[s], False)}
    if nq in own:
        continue
    others = [o for o in at_addr.get(key_of[s], []) if o != s and nq in {initials(names[o]), initials(names[o], False)}]
    B.append((q, s, others))
rep.write(f"\nB. accepted initials records not matching the S1's initials: {len(B)}; "
          f"with another S1 at the same address that fits: {sum(1 for b in B if b[2])}\n")
for q, s, o in B[:25]:
    rep.write(f"   {names[q]!r} -> {names[s]!r} | {addrs[s]!r}  other fits: {[names[x] for x in o]}\n")

# ---- C: unassigned web handles at an S1 address
C = []
for q in qs:
    if q in assigned:
        continue
    n = names[q].strip()
    if " " in n or len(n) < 6:
        continue
    h = asc(n)
    if not (h.endswith(".com") or h.endswith("com") or h.startswith("#") or h.startswith("@")):
        continue
    h = re.sub(r"(\.com|com)$", "", h).lstrip("#@").replace(".", "")
    k = key_of[q]
    for s in at_addr.get(k, []):
        full = "".join(t for t in toks(names[s]))
        nolegal = "".join(t for t in toks(names[s]) if t not in LEGAL)
        if max(fuzz.ratio(h, full), fuzz.ratio(h, nolegal)) >= 85 or (len(h) >= 6 and (full.startswith(h) or nolegal.startswith(h))):
            C.append((q, s))
            break
rep.write(f"\nC. unassigned web handles at an S1's exact address spelling that S1: {len(C)} "
          f"(S1s {len({c[1] for c in C})}, S1 has room: {sum(1 for q, s in C if nmatch[s] < 11)})\n")
for q, s in C[:25]:
    rep.write(f"   {names[q]!r} | {addrs[q]!r} -> {names[s]!r} (nmatch {nmatch[s]})\n")

# ---- D: unassigned generic swaps at the S1's exact address
D = []
for q in qs:
    if q in assigned or key_of[q][1] is None:
        continue
    cands = at_addr.get(key_of[q], [])
    if len(cands) != 1:
        continue
    s = cands[0]
    tq, ts = set(toks(names[q])) - STOP, set(toks(names[s])) - STOP
    add, rem = (tq - ts) - LEGAL, (ts - tq) - LEGAL
    if add and add <= GENERIC and len(rem) <= 1 and not (add & DECOYW) and (ts - rem - LEGAL) and \
            ((tq & LEGAL) == (ts & LEGAL) or not (tq & LEGAL) or not (ts & LEGAL)):
        D.append((q, s, " ".join(sorted(add)), " ".join(sorted(rem))))
rep.write(f"\nD. unassigned generic-word variants at a single-S1 exact address: {len(D)} (S1s {len({d[1] for d in D})})\n")
rep.write("   added words: " + str(Counter(d[2] for d in D).most_common(15)) + "\n")
for q, s, a, r in D[:30]:
    rep.write(f"   +{a} -{r}: {names[s]!r} -> {names[q]!r} | {addrs[q]!r} (nmatch {nmatch[s]})\n")
rep.close()
pl.DataFrame(A, schema=["q", "s1", "extra", "removed", "added"], orient="row").write_parquet(f"{OUT}/rules_A.parquet")
pl.DataFrame([(q, s) for q, s in C], schema=["q", "s1"], orient="row").write_parquet(f"{OUT}/rules_C.parquet")
pl.DataFrame(D, schema=["q", "s1", "added", "removed"], orient="row").write_parquet(f"{OUT}/rules_D.parquet")
print("done")

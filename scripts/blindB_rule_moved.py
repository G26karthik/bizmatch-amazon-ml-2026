"""Blind audit B: candidate France rule "same-name moved-house" -> hit list + cap-test precision bounds.

Hit = accepted France pair (fr14) whose record name key (legal forms dropped, accents/case/punct normalised, tokens
sorted) equals the S1's, on the S1's street, with a parsed house number different from the S1's and not at a decoy
offset (+1,2,3,4,5,7,9,11,13,21), where the S1 has k >= 1 other accepted records of the same source at its exact house.
Classes: exact (normalised full name incl. legal form identical) / noisy; typo (one-digit edit) / jump.
Precision bounds per (class, source, k) cell from the cap test over all such records (accepted or not):
floor = pooled rate at k >= 4 (non-copies), copy share = (rate - floor) / rate;
FP >= accepted - copies (model picks copies first), FP <= accepted * (1 - copy share) (acceptance blind to copy status).
Usage: python scripts/blindB_rule_moved.py <test_dir> <matching_results.tsv> <out_dir>
"""
import re
import sys
import unicodedata
from collections import Counter, defaultdict

import polars as pl

TEST, SUB, OUT = sys.argv[1:4]
LEGAL = {"sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "ei", "selarl", "scop"}
REGION = {"hauts", "de", "france", "nord", "pas", "calais", "nouvelle", "aquitaine", "gironde", "pays", "la", "loire",
          "atlantique", "du", "des", "le", "les", "d", "l"}
ABBR = {"rue": "r", "avenue": "av", "ave": "av", "boulevard": "bd", "blvd": "bd", "bld": "bd", "blrd": "bd",
        "chemin": "ch", "place": "pl", "route": "rte", "impasse": "imp", "allee": "all", "quai": "q", "cours": "crs",
        "passage": "psg", "pass": "psg", "residence": "res", "saint": "st", "sainte": "ste"}
OFFS = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}
HN = re.compile(r"(?:^|[\s,#(]|n[o°º]\.?\s?)0*(\d{1,4})(?:\s?(?:bis|ter|[a-d])\b)?(?=[\s,)]|$)", re.I)


def asc(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


def ntok(name):
    return [t for t in re.split(r"[^a-z0-9]+", asc(name).replace(".", "")) if t]


def k1(name):
    return " ".join(sorted({t for t in ntok(name) if t not in LEGAL}))


def house(a):
    for m in HN.finditer(" " + asc(a)):
        return int(m.group(1))
    return None


def street(a):
    ts = [ABBR.get(t, t) for t in re.split(r"[^a-z0-9]+", asc(a)) if t and not any(c.isdigit() for c in t)]
    return " ".join(sorted({t for t in ts if len(t) > 1 and t not in REGION and t not in ("bis", "ter", "no")}))


def typo(hs, hq):
    a, b = str(hs), str(hq)
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1 or sorted(a) == sorted(b)
    if abs(len(a) - len(b)) == 1:
        lo, hi = (a, b) if len(a) < len(b) else (b, a)
        return any(hi[:i] + hi[i + 1:] == lo for i in range(len(hi)))
    return False


rec = pl.concat([pl.read_csv(f"{TEST}/test_source{k}.tsv", separator="\t", quote_char=None, infer_schema_length=0)
                 .filter(pl.col("country") == "France").with_columns(pl.lit(k).alias("src")) for k in (1, 2, 3)])
rows = list(rec.select("entity_id", "business_name", "business_address", "src").iter_rows())
del rec
mr = pl.read_csv(SUB, separator="\t", infer_schema_length=0).with_columns(pl.col("matched_entity_ids").fill_null(""))
links = {q: s for s, m in mr.iter_rows() for q in m.split(",") if q}
s1, idx = {}, defaultdict(list)
for e, n, a, src in rows:
    if src == 1:
        h = house(a)
        if h is not None:
            s1[e] = (h, n, a)
            idx[(k1(n), street(a))].append(e)
nsrc = defaultdict(lambda: [0, 0, 0, 0])
moved = []
for e, n, a, src in rows:
    if src == 1 or not a:
        continue
    hq = house(a)
    s = links.get(e)
    if s in s1 and hq is not None and s1[s][0] == hq:
        nsrc[s][src] += 1
    if hq is None:
        continue
    for s in idx.get((k1(n), street(a)), []):
        hs, sn, sa = s1[s]
        if hq != hs and (hq - hs) not in OFFS:
            exact = ntok(n) == ntok(sn)
            cls = ("exact" if exact else "noisy") + "/" + ("typo" if typo(hs, hq) else "jump")
            moved.append((s, e, src, cls, links.get(e) == s, n, a))
nS = Counter()
for s in s1:
    for src in (2, 3):
        nS[(src, min(nsrc[s][src], 5 if src == 2 else 6))] += 1
cell = Counter()
for s, e, src, cls, acc, n, a in moved:
    k = min(nsrc[s][src], 5 if src == 2 else 6)
    cell[(cls, src, k, "n")] += 1
    cell[(cls, src, k, "acc")] += acc
rep = open(f"{OUT}/rule_moved_report.txt", "w", encoding="utf-8")
hits = []
tot = Counter()
for cls in ("exact/jump", "exact/typo", "noisy/jump", "noisy/typo"):
    for src in (2, 3):
        cap = 5 if src == 2 else 6
        fl_n = sum(cell[(cls, src, k, "n")] for k in range(4, cap + 1))
        fl_d = sum(nS[(src, k)] for k in range(4, cap + 1))
        floor = fl_n / max(fl_d, 1)
        for k in range(1, cap + 1):
            n, acc, d = cell[(cls, src, k, "n")], cell[(cls, src, k, "acc")], nS[(src, k)]
            if not n:
                continue
            rate = n / d
            cs = max(0.0, (rate - floor) / rate)
            copies = cs * n
            lb, ub = max(0.0, acc - copies), acc * (1 - cs)
            rep.write(f"{cls:11s} S{src} k={k}: rate {1000 * rate:6.2f}/1000 floor {1000 * floor:5.2f} copy share "
                      f"{cs:.2f} | accepted {acc:5d} FP in [{lb:6.0f}, {ub:6.0f}]\n")
            tot[("acc", cls, k >= 2)] += acc
            tot[("lb", cls, k >= 2)] += lb
            tot[("ub", cls, k >= 2)] += ub
for s, e, src, cls, acc, n, a in moved:
    k = nsrc[s][src]
    if acc and k >= 1:
        hits.append((s, e, src, cls, k, s1[s][1], s1[s][2], n, a))
rep.write("\nTotals (accepted hits, FP lower/upper bound) by class and k>=2 vs k=1:\n")
for cls in ("exact/jump", "exact/typo", "noisy/jump", "noisy/typo"):
    for g in (False, True):
        rep.write(f"  {cls:11s} {'k>=2' if g else 'k=1 '}: accepted {tot[('acc', cls, g)]:5.0f}  FP in "
                  f"[{tot[('lb', cls, g)]:5.0f}, {tot[('ub', cls, g)]:5.0f}]\n")
df = pl.DataFrame(hits, schema=["s1", "q", "src", "cls", "k", "s1_name", "s1_addr", "q_name", "q_addr"], orient="row")
df.write_parquet(f"{OUT}/rule_moved_hits.parquet")
rep.write(f"\nhit list: {df.height} pairs, {df['s1'].n_unique()} S1s (k>=1); k>=2: "
          f"{df.filter(pl.col('k') >= 2).height} pairs, {df.filter(pl.col('k') >= 2)['s1'].n_unique()} S1s\n")
for r in df.filter(pl.col("k") >= 2).sample(min(25, df.height), seed=7).iter_rows():
    rep.write(f"  [{r[3]} S{r[2]} k={r[4]}] {r[5]!r} | {r[6]!r}  ->  {r[7]!r} | {r[8]!r}\n")
rep.close()
print("done")

"""Blind audit B: moved-house same-name records, cap test split by house relation class and cluster membership.

class: typo (one digit substituted / dropped / added / transposed) vs jump (any other non-offset house);
cluster: >= 2 records of the same S1 name key at the same moved house vs single.
Usage: python scripts/blindB_moved2.py test <test_dir> <matching_results.tsv> <out>   (France/US/India, accepted share)
       python scripts/blindB_moved2.py train <train_dir> <ground_truth.tsv> <out>     (US/India, truth share)
"""
import re
import sys
import unicodedata
from collections import Counter, defaultdict

import polars as pl

MODE, D, LINKS, OUT = sys.argv[1:5]
LEGAL = {"sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "ei", "selarl", "scop", "inc", "llc", "corp", "co", "ltd",
         "corporation", "incorporated", "pllc", "lp", "llp", "pvt", "private", "limited", "opc", "company"}
REGION = {"hauts", "de", "france", "nord", "pas", "calais", "nouvelle", "aquitaine", "gironde", "pays", "la", "loire",
          "atlantique", "du", "des", "le", "les", "d", "l"}
ABBR = {"rue": "r", "avenue": "av", "ave": "av", "boulevard": "bd", "blvd": "bd", "bld": "bd", "blrd": "bd",
        "chemin": "ch", "place": "pl", "route": "rte", "impasse": "imp", "allee": "all", "quai": "q", "cours": "crs",
        "passage": "psg", "pass": "psg", "residence": "res", "saint": "st", "sainte": "ste", "road": "rd",
        "street": "st", "drive": "dr", "lane": "ln", "court": "ct", "highway": "hwy", "square": "sq"}
OFFS = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}
HN = re.compile(r"(?:^|[\s,#(]|n[o°º]\.?\s?)0*(\d{1,4})(?:\s?(?:bis|ter|[a-d])\b)?(?=[\s,)]|$)", re.I)


def asc(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


def k1(name):
    return " ".join(sorted({t for t in re.split(r"[^a-z0-9]+", asc(name).replace(".", "")) if t and t not in LEGAL}))


def house(a):
    for m in HN.finditer(" " + asc(a)):
        return int(m.group(1))
    return None


def street(a):
    toks = [ABBR.get(t, t) for t in re.split(r"[^a-z0-9]+", asc(a)) if t and not any(c.isdigit() for c in t)]
    return " ".join(sorted({t for t in toks if len(t) > 1 and t not in REGION and t not in ("bis", "ter", "no")}))


def typo(hs, hq):
    a, b = str(hs), str(hq)
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1 or (sorted(a) == sorted(b))
    if abs(len(a) - len(b)) == 1:
        lo, hi = (a, b) if len(a) < len(b) else (b, a)
        return any(hi[:i] + hi[i + 1:] == lo for i in range(len(hi)))
    return False


pre = "test" if MODE == "test" else "train"
KEEP = ["France"] if MODE == "test" else ["US"]
rec = pl.concat([pl.read_csv(f"{D}/{pre}_source{k}.tsv", separator="\t", quote_char=None, infer_schema_length=0)
                 .filter(pl.col("country").is_in(KEEP)).with_columns(pl.lit(k).alias("src")) for k in (1, 2, 3)])
lk = pl.read_csv(LINKS, separator="\t", infer_schema_length=0).with_columns(pl.col("matched_entity_ids").fill_null(""))
links = {q: s for s, m in lk.iter_rows() for q in m.split(",") if q}
s1 = {}
idx = defaultdict(list)
rows = list(rec.select("entity_id", "business_name", "business_address", "country", "src").iter_rows())
del rec
for e, n, a, c, src in rows:
    if src == 1:
        h = house(a)
        if h is None:
            continue
        key = (c, k1(n), street(a))
        s1[e] = (h, c)
        idx[key].append(e)
nsrc = defaultdict(lambda: [0, 0, 0, 0])
moved = []
for e, n, a, c, src in rows:
    if src == 1 or not a:
        continue
    hq = house(a)
    s = links.get(e)
    if s in s1 and hq is not None and s1[s][0] == hq:
        nsrc[s][src] += 1
    if hq is None:
        continue
    for s in idx.get((c, k1(n), street(a)), []):
        hs = s1[s][0]
        if hq != hs and (hq - hs) not in OFFS:
            moved.append((s, e, src, c, hq, "typo" if typo(hs, hq) else "jump", links.get(e) == s))
cl = Counter((s, hq) for s, e, src, c, hq, t, acc in moved)
nS = Counter()
for s, (h, c) in s1.items():
    for src in (2, 3):
        nS[(c, src, min(nsrc[s][src], 5 if src == 2 else 6))] += 1
agg = Counter()
for s, e, src, c, hq, t, acc in moved:
    k = min(nsrc[s][src], 5 if src == 2 else 6)
    grp = f"{t}/{'clu' if cl[(s, hq)] >= 2 else 'one'}"
    agg[(c, src, k, grp, "n")] += 1
    agg[(c, src, k, grp, "acc")] += acc
with open(OUT, "w", encoding="utf-8") as f:
    f.write(f"[{MODE}] moved-house same-name same-street records per 1000 S1, by n exact-house linked records k; "
            f"share = {'accepted' if MODE == 'test' else 'true'}\n")
    for c in sorted({c for _, (h, c) in s1.items()}):
        for src in (2, 3):
            cap = 5 if src == 2 else 6
            for k in range(cap + 1):
                n = nS[(c, src, k)]
                if n < 50:
                    continue
                parts = []
                for g in ("typo/one", "typo/clu", "jump/one", "jump/clu"):
                    x = agg[(c, src, k, g, "n")]
                    parts.append(f"{g} {1000 * x / n:6.2f} ({agg[(c, src, k, g, 'acc')] / max(x, 1):.2f})")
                f.write(f"  {c} S{src} k={k} nS1={n:7d}: " + " | ".join(parts) + "\n")
print("done")

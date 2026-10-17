"""Blind audit B: France multi-word type phrase swaps ("École maternelle X" -> "Centre X") at the S1's exact address.

Cap test: rate per 1000 S1 (S1 name starting with the phrase) of such swap records by the S1's accepted exact-house
count k in that source, with the accepted share; single-word vocabulary swaps (the team's vocabulary rule class) and
generic-word swaps (known copies) as references.
Usage: python scripts/blindB_multiword.py <test_dir> <matching_results.tsv> <out>
"""
import re
import sys
import unicodedata
from collections import Counter, defaultdict

import polars as pl

TEST, SUB, OUT = sys.argv[1:4]
LEGAL = {"sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "ei", "selarl", "scop"}
STOP = {"de", "du", "des", "la", "le", "les", "l", "d", "et", "en", "a", "au", "aux"}
GENERIC = {"fils", "compagnie", "cie", "associes", "services", "developpement", "groupe", "france", "freres", "frs"}
PHRASES = [("ecole", "maternelle"), ("ecole", "primaire"), ("centre", "hospitalier"), ("centre", "medical"),
           ("maison", "sante")]
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


rec = pl.concat([pl.read_csv(f"{TEST}/test_source{k}.tsv", separator="\t", quote_char=None, infer_schema_length=0)
                 .filter(pl.col("country") == "France").with_columns(pl.lit(k).alias("src")) for k in (1, 2, 3)])
rows = list(rec.select("entity_id", "business_name", "business_address", "src").iter_rows())
del rec
mr = pl.read_csv(SUB, separator="\t", infer_schema_length=0).with_columns(pl.col("matched_entity_ids").fill_null(""))
links = {q: s for s, m in mr.iter_rows() for q in m.split(",") if q}
s1 = {}
at = defaultdict(list)
df = Counter()
for e, n, a, src in rows:
    if src == 1:
        k = (street(a), house(a))
        s1[e] = (n, k)
        if k[1] is not None and k[0]:
            at[k].append(e)
        for t in set(toks(n)):
            df[t] += 1
VOC = {t for t, c in df.items() if c >= 150 and t not in LEGAL and t not in STOP and t not in GENERIC and len(t) > 2}
nsrc = defaultdict(lambda: [0, 0, 0, 0])
cls_rows = []
for e, n, a, src in rows:
    if src == 1:
        continue
    k = (street(a), house(a))
    s = links.get(e)
    if s in s1 and k[1] is not None and s1[s][1] == k:
        nsrc[s][src] += 1
    cands = at.get(k, [])
    if len(cands) != 1:
        continue
    s = cands[0]
    ts, tq = [t for t in toks(s1[s][0]) if t not in STOP], [t for t in toks(n) if t not in STOP]
    rem, add = set(ts) - set(tq) - LEGAL, set(tq) - set(ts) - LEGAL
    if not rem or not add or len(add) != 1:
        continue
    cls = None
    if any(set(p) == rem and tuple(ts[:2]) == p for p in PHRASES) and add <= VOC:
        cls = "multiword->vocab"
    elif len(rem) == 1 and rem <= VOC and add <= VOC and not any(a2[:3] == r[:3] for a2 in add for r in rem):
        cls = "vocab->vocab"
    elif len(rem) == 1 and add <= GENERIC:
        cls = "type->generic"
    if cls:
        cls_rows.append((s, e, src, cls, links.get(e) == s))
nS = Counter()
starts = {s for s, (n, k) in s1.items() if any(tuple([t for t in toks(n) if t not in STOP][:2]) == p for p in PHRASES)}
for s in s1:
    for src in (2, 3):
        for grp in (("multi", s in starts), ("all", True)):
            if grp[1]:
                nS[(grp[0], src, min(nsrc[s][src], 5 if src == 2 else 6))] += 1
agg = Counter()
for s, e, src, cls, acc in cls_rows:
    k = min(nsrc[s][src], 5 if src == 2 else 6)
    agg[(cls, src, k, "n")] += 1
    agg[(cls, src, k, "acc")] += acc
with open(OUT, "w", encoding="utf-8") as f:
    f.write(f"France same-address single-S1 swaps; per 1000 S1 (multiword: S1s whose name starts with the phrase, "
            f"n={len(starts)}; others: all S1s), accepted share in ()\n")
    for cls in ("multiword->vocab", "vocab->vocab", "type->generic"):
        base = "multi" if cls.startswith("multi") else "all"
        tot = sum(agg[(cls, src, k, "n")] for src in (2, 3) for k in range(7))
        acc = sum(agg[(cls, src, k, "acc")] for src in (2, 3) for k in range(7))
        f.write(f"{cls}: records {tot}, accepted {acc}\n")
        for src in (2, 3):
            parts = []
            for k in range(7):
                n = nS[(base, src, k)]
                if n >= 30:
                    x = agg[(cls, src, k, "n")]
                    parts.append(f"k={k} {1000 * x / n:6.2f} ({agg[(cls, src, k, 'acc')] / max(x, 1):.2f})")
            f.write(f"   S{src}: " + " | ".join(parts) + "\n")
print("done")

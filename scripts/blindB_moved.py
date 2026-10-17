"""Blind audit B: are France same-name / same-street / moved-house records copies or decoys?  Cap test.

For each S1: n_src = number of submission-accepted records of that source at the S1's exact house (sure copies).
For each S2/S3 record with the S1's name key (legal forms dropped, sorted tokens) on the S1's street key:
relation = exact | offset (+1,2,3,4,5,7,9,11,13,21) | moved (any other house) | nohouse.
Rate of relation-r records per S1, split by n_src. Copies must vanish at n_src = cap (5 for S2, 6 for S3); decoys do not.
Train (US/India truth) gives the same table with true/false split as a calibration.
Usage: python scripts/blindB_moved.py <test_dir> <train_dir> <matching_results.tsv> <out_dir>
"""
import re
import sys
import unicodedata
from collections import Counter, defaultdict

import polars as pl

TEST, TRAIN, SUB, OUT = sys.argv[1:5]
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
    s = asc(name).replace(".", "")
    return " ".join(sorted({t for t in re.split(r"[^a-z0-9]+", s) if t and t not in LEGAL}))


def house(a):
    for m in HN.finditer(" " + asc(a)):
        return int(m.group(1))
    return None


def street(a):
    toks = [ABBR.get(t, t) for t in re.split(r"[^a-z0-9]+", asc(a)) if t and not any(c.isdigit() for c in t)]
    return " ".join(sorted({t for t in toks if len(t) > 1 and t not in REGION and t not in ("bis", "ter", "no")}))


def load(d, pre):
    dfs = []
    for k in (1, 2, 3):
        df = pl.read_csv(f"{d}/{pre}_source{k}.tsv", separator="\t", quote_char=None, infer_schema_length=0)
        dfs.append(df.with_columns(pl.lit(k).alias("src")))
    return pl.concat(dfs)


def analyse(rec, links, countries, tag, truth=None):
    """links: q -> s1 (accepted or true); returns report lines."""
    s1 = {}
    idx = defaultdict(list)
    for e, n, a, c, src in rec.select("entity_id", "business_name", "business_address", "country", "src").iter_rows():
        if src == 1 and c in countries:
            key = (c, k1(n), street(a))
            s1[e] = (house(a), key)
            idx[key].append(e)
    nsrc = defaultdict(lambda: [0, 0, 0, 0])
    rel = []
    for e, n, a, c, src in rec.select("entity_id", "business_name", "business_address", "country", "src").iter_rows():
        if src == 1 or c not in countries:
            continue
        hq = house(a)
        s = links.get(e)
        if s in s1 and hq is not None and s1[s][0] == hq:
            nsrc[s][src] += 1
        key = (c, k1(n), street(a))
        cands = idx.get(key)
        if not cands or not a:
            continue
        for s in cands:
            hs = s1[s][0]
            if hs is None or hq is None:
                r = "nohouse"
            elif hq == hs:
                r = "exact"
            elif hq - hs in OFFS:
                r = "offset"
            else:
                r = "moved"
            rel.append((s, e, src, r, c, links.get(e) == s, None if truth is None else truth.get(e) == s))
    out = []
    for c in sorted(countries):
        n_s1 = Counter()
        for s, (h, key) in s1.items():
            if key[0] == c and h is not None:
                for src in (2, 3):
                    n_s1[(src, min(nsrc[s][src], 6 if src == 3 else 5))] += 1
        agg = Counter()
        for s, e, src, r, cc, acc, tru in rel:
            if cc != c or s1[s][0] is None:
                continue
            k = min(nsrc[s][src], 6 if src == 3 else 5)
            agg[(src, k, r, "all")] += 1
            if acc:
                agg[(src, k, r, "acc")] += 1
            if tru:
                agg[(src, k, r, "true")] += 1
        out.append(f"[{tag}] {c}: rate per 1000 S1 of same-name same-street records by relation, split by n exact-house "
                   f"accepted records of that source (cap S2=5, S3=6); (acc = accepted share, true = truth share)")
        for src in (2, 3):
            cap = 5 if src == 2 else 6
            for k in range(cap + 1):
                n = n_s1[(src, k)]
                if n == 0:
                    continue
                parts = []
                for r in ("moved", "offset", "nohouse"):
                    a = agg[(src, k, r, "all")]
                    parts.append(f"{r} {1000 * a / n:7.2f} (acc {agg[(src, k, r, 'acc')] / max(a, 1):.2f}"
                                 + (f", true {agg[(src, k, r, 'true')] / max(a, 1):.2f}" if truth is not None else "") + ")")
                out.append(f"   S{src} k={k} nS1={n:7d}: " + " | ".join(parts))
    return out, rel


rep = open(f"{OUT}/moved_report.txt", "w", encoding="utf-8")
te = load(TEST, "test")
mr = pl.read_csv(SUB, separator="\t", infer_schema_length=0).with_columns(pl.col("matched_entity_ids").fill_null(""))
links = {q: s for s, m in mr.iter_rows() for q in m.split(",") if q}
lines, rel = analyse(te, links, {"France", "US", "India"}, "test fr14")
rep.write("\n".join(lines) + "\n")
pl.DataFrame([r[:6] for r in rel if r[3] == "moved"], schema=["s1", "q", "src", "rel", "country", "accepted"],
             orient="row").write_parquet(f"{OUT}/moved_pairs_test.parquet")
rep.flush()
del te
tr = load(TRAIN, "train")
gt = pl.read_csv(f"{TRAIN}/train_ground_truth.tsv", separator="\t", infer_schema_length=0).with_columns(
    pl.col("matched_entity_ids").fill_null(""))
truth = {q: s for s, m in gt.iter_rows() for q in m.split(",") if q}
lines, _ = analyse(tr, truth, {"US", "India"}, "train truth", truth)
rep.write("\n".join(lines) + "\n")
rep.close()
print("done")

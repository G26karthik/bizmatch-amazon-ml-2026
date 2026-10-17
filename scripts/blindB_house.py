"""Blind audit B: house-number differences inside accepted pairs (France vs US/India) and in train truth.

Usage: python scripts/blindB_house.py <test_dir> <train_dir> <matching_results.tsv> <out_dir>
"""
import re
import sys
import unicodedata
from collections import Counter

import polars as pl

TEST, TRAIN, SUB, OUT = sys.argv[1:5]
HN = re.compile(r"(?:^|[\s,#(]|n[o°º]\.?\s?)0*(\d{1,4})(?:\s?(?:bis|ter|[a-d])\b)?(?=[\s,)]|$)", re.I)


def house(a):
    if not a:
        return None
    s = unicodedata.normalize("NFKD", a).encode("ascii", "ignore").decode().lower()
    for m in HN.finditer(" " + s):
        return int(m.group(1))
    return None


def load(d, pre, k):
    return pl.read_csv(f"{d}/{pre}_source{k}.tsv", separator="\t", quote_char=None, infer_schema_length=0)


def diffs(pairs, rec):
    """pairs: (s1_id, q_id) -> list of (country, diff, s1_house, q_house, names, addrs)"""
    out = []
    for s, q in pairs:
        a, b = rec.get(s), rec.get(q)
        if a is None or b is None:
            continue
        hs, hq = house(a[1]), house(b[1])
        out.append((a[2], None if hs is None or hq is None else hq - hs, hs, hq, a, b))
    return out


def summarize(tag, rows, f):
    by = {}
    for c, d, hs, hq, a, b in rows:
        by.setdefault(c, []).append(d)
    for c, ds in sorted(by.items()):
        known = [d for d in ds if d is not None]
        cnt = Counter(known)
        n = len(known)
        f.write(f"[{tag}] {c}: pairs {len(ds)}, both houses {n}, equal {cnt[0]} ({cnt[0] / max(n, 1):.4f}), "
                f"diff!=0 {n - cnt[0]} ({(n - cnt[0]) / max(n, 1):.4f})\n")
        top = [(d, k) for d, k in cnt.most_common(40) if d != 0]
        f.write("   top diffs: " + " ".join(f"{d:+d}:{k}" for d, k in top[:30]) + "\n")


with open(f"{OUT}/house_report.txt", "w", encoding="utf-8") as f:
    # test: accepted pairs
    trec = {}
    for k in (1, 2, 3):
        df = load(TEST, "test", k)
        for e, n, a, c in df.select("entity_id", "business_name", "business_address", "country").iter_rows():
            trec[e] = (n or "", a or "", c)
    mr = pl.read_csv(SUB, separator="\t", infer_schema_length=0).with_columns(pl.col("matched_entity_ids").fill_null(""))
    pairs = [(s, q) for s, m in mr.iter_rows() for q in m.split(",") if q]
    rows = diffs(pairs, trec)
    summarize("test accepted", rows, f)
    fr = [r for r in rows if r[0] == "France" and r[1] not in (None, 0)]
    f.write(f"\nFrance accepted pairs with house diff != 0: {len(fr)}; S1s affected {len({r[4] for r in fr})}\n")
    for r in fr[:150]:
        f.write(f"  {r[1]:+5d} | {r[4][0]} | {r[4][1]}  ||  {r[5][0]} | {r[5][1]}\n")
    pl.DataFrame([(r[4][0], r[4][1], r[5][0], r[5][1], r[1]) for r in fr],
                 schema=["s1_name", "s1_addr", "q_name", "q_addr", "diff"], orient="row").write_parquet(f"{OUT}/fr_house_diff.parquet")
    # train truth
    rrec = {}
    for k in (1, 2, 3):
        df = load(TRAIN, "train", k)
        for e, n, a, c in df.select("entity_id", "business_name", "business_address", "country").iter_rows():
            rrec[e] = (n or "", a or "", c)
    gt = pl.read_csv(f"{TRAIN}/train_ground_truth.tsv", separator="\t", infer_schema_length=0)
    cols = gt.columns
    f.write(f"\ntrain gt columns {cols}\n")
    if len(cols) == 2 and "matched" in cols[1]:
        gpairs = [(s, q) for s, m in gt.iter_rows() for q in (m or "").split(",") if q]
    else:
        gpairs = list(gt.select(cols[0], cols[1]).iter_rows())
    summarize("train truth", diffs(gpairs, rrec), f)
print("done")

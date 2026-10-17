"""Blind audit B: empty-address records whose name uniquely matches one S1 (same country, name key without legal forms).

TRAIN (US truth): share of such records that are true copies of that S1, split by whether the S1 has a same-name
record at a decoy offset / moved house on its street ("decoy cluster present").
TEST (France, US): the same records by submission state; cap test of the unassigned ones (rate per S1 by the S1's
accepted count in the record's source; copies must vanish at the cap).
Usage: python scripts/blindB_empty.py <test_dir> <train_dir> <matching_results.tsv> <out_dir>
"""
import re
import sys
import unicodedata
from collections import Counter, defaultdict

import polars as pl

TEST, TRAIN, SUB, OUT = sys.argv[1:5]
LEGAL = {"sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "ei", "selarl", "scop", "inc", "llc", "corp", "co", "ltd",
         "corporation", "incorporated", "pllc", "lp", "llp", "pvt", "private", "limited", "opc", "company"}
GENERIC = {"fils", "compagnie", "associes", "services", "service", "developpement", "groupe", "france", "freres",
           "center", "partners"}
MAP = {"cie": "compagnie", "frs": "freres", "et": "", "and": "", "the": ""}


def toks(name):
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower().replace(".", "")
    return [MAP.get(t, t) for t in re.split(r"[^a-z0-9]+", s) if t and MAP.get(t, t) and t not in LEGAL]


def k1(name):
    return " ".join(sorted(set(toks(name))))


def k2(name):
    return " ".join(sorted(set(t for t in toks(name) if t not in GENERIC)))


def load(d, pre, countries):
    out = []
    for k in (1, 2, 3):
        df = pl.read_csv(f"{d}/{pre}_source{k}.tsv", separator="\t", quote_char=None, infer_schema_length=0,
                         columns=["entity_id", "business_name", "business_address", "country"])
        df = df.filter(pl.col("country").is_in(countries)).with_columns(
            pl.lit(k).alias("src"), (pl.col("business_address").fill_null("").str.strip_chars() == "").alias("empty"))
        out.append(df.filter((pl.col("src") == 1) | pl.col("empty") | pl.lit(k > 1)))
    return pl.concat(out)


def build(rec):
    s1 = rec.filter(pl.col("src") == 1)
    i1, i2 = defaultdict(list), defaultdict(list)
    for e, n, c in s1.select("entity_id", "business_name", "country").iter_rows():
        i1[(c, k1(n))].append(e)
        i2[(c, k2(n))].append(e)
    # same-name records WITH an address (possible decoy clusters): name key -> count of non-empty records
    ne = Counter()
    for n, c in rec.filter((pl.col("src") > 1) & ~pl.col("empty")).select("business_name", "country").iter_rows():
        ne[(c, k1(n))] += 1
    return i1, i2, ne


def candidates(rec, i1, i2):
    out = []
    for e, n, c, src in rec.filter((pl.col("src") > 1) & pl.col("empty")).select(
            "entity_id", "business_name", "country", "src").iter_rows():
        a, b = i1.get((c, k1(n)), []), i2.get((c, k2(n)), [])
        if len(a) == 1:
            out.append((e, c, src, "K1", a[0], n))
        elif not a and len(b) == 1 and k2(n):
            out.append((e, c, src, "K2", b[0], n))
    return out


rep = open(f"{OUT}/empty_report.txt", "w", encoding="utf-8")
# ---- train truth (US)
tr = load(TRAIN, "train", ["US"])
emp = set(tr.filter(pl.col("empty") & (pl.col("src") > 1))["entity_id"].to_list())
gt = pl.read_csv(f"{TRAIN}/train_ground_truth.tsv", separator="\t", infer_schema_length=0).with_columns(
    pl.col("matched_entity_ids").fill_null(""))
truth, tcnt = {}, defaultdict(lambda: [0, 0, 0, 0])
for s, m in gt.iter_rows():
    for q in m.split(","):
        if q in emp:
            truth[q] = s
i1, i2, ne = build(tr)
s1name = dict(zip(tr.filter(pl.col("src") == 1)["entity_id"].to_list(), tr.filter(pl.col("src") == 1)["business_name"].to_list()))
cand = candidates(tr, i1, i2)
st = Counter()
for e, c, src, kind, s1, n in cand:
    t = truth.get(e)
    dec = "sameName-addr-records>1" if ne[(c, k1(s1name[s1]))] > 1 else "few"
    st[(kind, dec, "true" if t == s1 else ("other" if t else "nonmatch"))] += 1
rep.write("TRAIN US truth: empty-address records with a unique name-key S1 match\n")
for k in sorted(st):
    rep.write(f"  {k}: {st[k]}\n")
del tr, gt, truth
# ---- test
te = load(TEST, "test", ["France", "US"])
mr = pl.read_csv(SUB, separator="\t", infer_schema_length=0).with_columns(pl.col("matched_entity_ids").fill_null(""))
s1set = set(te.filter(pl.col("src") == 1)["entity_id"].to_list())
assigned, cnt = {}, defaultdict(lambda: [0, 0, 0, 0])
for s, m in mr.iter_rows():
    if s not in s1set:
        continue
    for q in m.split(","):
        if q:
            assigned[q] = s
            cnt[s][2 if q.startswith("S2") else 3] += 1
i1, i2, ne = build(te)
s1name = dict(zip(te.filter(pl.col("src") == 1)["entity_id"].to_list(), te.filter(pl.col("src") == 1)["business_name"].to_list()))
s1addr = dict(zip(te.filter(pl.col("src") == 1)["entity_id"].to_list(), te.filter(pl.col("src") == 1)["business_address"].to_list()))
s1c = dict(zip(te.filter(pl.col("src") == 1)["entity_id"].to_list(), te.filter(pl.col("src") == 1)["country"].to_list()))
cand = candidates(te, i1, i2)
st = Counter()
rows = []
for e, c, src, kind, s1, n in cand:
    a = assigned.get(e)
    state = "same" if a == s1 else ("other" if a else "unassigned")
    dec = "sameName-addr-records>1" if ne[(c, k1(s1name[s1]))] > 1 else "few"
    st[(c, kind, dec, state)] += 1
    if state == "unassigned":
        rows.append((c, kind, dec, e, n, s1, s1name[s1], s1addr[s1] or "", cnt[s1][2] + cnt[s1][3], src, cnt[s1][src]))
rep.write("\nTEST fr14: empty-address records with a unique name-key S1 match (country, key, decoy-name-present, state)\n")
for k in sorted(st):
    rep.write(f"  {k}: {st[k]}\n")
nk, hk = Counter(), Counter()
for s, c in s1c.items():
    for src in (2, 3):
        nk[(c, src, min(cnt[s][src], 5 if src == 2 else 6))] += 1
for e, c, src, kind, s1, n in cand:
    a = assigned.get(e)
    state = "same" if a == s1 else ("other" if a else "unassigned")
    hk[(c, src, min(cnt[s1][src], 5 if src == 2 else 6), state)] += 1
rep.write("\nCap test (per 1000 S1): unassigned / accepted-same hits by the S1's accepted count k in that source\n")
for c in ("France", "US"):
    for src in (2, 3):
        cap = 5 if src == 2 else 6
        row = []
        for k in range(cap + 1):
            n = nk[(c, src, k)]
            if n:
                row.append(f"k={k}: {1000 * hk[(c, src, k, 'unassigned')] / n:6.2f}/{1000 * hk[(c, src, k, 'same')] / n:6.2f} (n={n})")
        rep.write(f"  {c} S{src}: " + " | ".join(row) + "\n")
df = pl.DataFrame(rows, schema=["country", "kind", "decoyname", "q", "q_name", "s1", "s1_name", "s1_addr", "s1_nmatch",
                                "src", "s1_nsrc"], orient="row")
df.write_parquet(f"{OUT}/empty_unassigned.parquet")
rep.write("\nsample unassigned France hits:\n")
for r in df.filter(pl.col("country") == "France").sample(min(40, df.height), seed=1).iter_rows():
    rep.write(f"  {r[1]} {r[2]} {r[4]!r} -> {r[6]!r} | {r[7]!r} nmatch={r[8]}\n")
rep.close()
print("done")

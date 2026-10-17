"""Blind audit B (local fallback, raw test TSVs): compact neighbourhoods for a random S1 sample.

Usage: python scripts/blindB_local.py <test_dir> <matching_results.tsv> <candidate_pairs.tsv> <out_dir>
Tags: M = matched to this S1; c = in this S1's candidate list, not matched to it; n = similar record (not a
candidate); S = other S1 nearby. Records assigned to another S1 show that S1 after '->'.
"""
import os
import re
import sys
import time
import unicodedata

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

TEST, SUB, CAND, OUT = sys.argv[1:5]
SEED = 20260927
NS = {"France": 250, "US": 100, "India": 60}
t0 = time.time()

ABBR = {
    "rue": "r", "avenue": "av", "ave": "av", "boulevard": "bd", "blvd": "bd", "bld": "bd", "chemin": "ch",
    "place": "pl", "route": "rte", "impasse": "imp", "allee": "all", "quai": "q", "cours": "crs",
    "passage": "psg", "pass": "psg", "residence": "res", "saint": "st", "sainte": "ste", "road": "rd",
    "street": "st", "drive": "dr", "lane": "ln", "court": "ct", "highway": "hwy", "suite": "ste", "square": "sq",
    "faubourg": "fbg", "no": "", "n": "", "bis": "", "ter": "",
}


def norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def addr_w(s):
    toks = [ABBR.get(t, t) for t in norm(s).split() if not any(ch.isdigit() for ch in t)]
    return " ".join(t for t in toks if t)


def house(s):
    m = re.search(r"\d+", s or "")
    return str(int(m.group())) if m else ""


def load(p, src):
    return pl.read_csv(p, separator="\t", quote_char=None, infer_schema_length=0).with_columns(
        pl.lit(src).alias("src"))


rec = pl.concat([load(f"{TEST}/test_source{k}.tsv", k) for k in (1, 2, 3)]).with_row_index("i")
eids = rec["entity_id"].to_list()
names = [x or "" for x in rec["business_name"].to_list()]
addrs = [x or "" for x in rec["business_address"].to_list()]
ctry = rec["country"].to_list()
srcs = rec["src"].to_list()
eid2i = {e: i for i, e in enumerate(eids)}
print("records", rec.height, time.time() - t0, flush=True)

mr = pl.read_csv(SUB, separator="\t", infer_schema_length=0).with_columns(pl.col("matched_entity_ids").fill_null(""))
assigned, s1_matches = {}, {}
for s, m in zip(mr["source1_entity_id"].to_list(), mr["matched_entity_ids"].to_list()):
    si = eid2i[s]
    lst = [eid2i[x] for x in m.split(",") if x]
    s1_matches[si] = lst
    for q in lst:
        assigned[q] = si
cp = pl.read_csv(CAND, separator="\t", infer_schema_length=0).with_columns(pl.col("candidate_entity_ids").fill_null(""))
cand_of = dict(zip(cp["source1_entity_id"].to_list(), cp["candidate_entity_ids"].to_list()))
print("submission", len(assigned), time.time() - t0, flush=True)

rng = np.random.default_rng(SEED)
sample = []
for c, n in NS.items():
    ids = np.array([i for i in range(rec.height) if srcs[i] == 1 and ctry[i] == c])
    sample += [(c, int(x)) for x in rng.choice(ids, n, replace=False)]


def short(x, n):
    x = x.replace("\t", " ")
    return x if len(x) <= n else x[: n - 1] + "~"


def line(tag, q, this):
    a = assigned.get(q)
    tail = "-> none" if a is None else ("" if a == this else f"-> {eids[a]} [{short(names[a], 36)} | {short(addrs[a], 50)}]")
    return f" {tag} {eids[q]:<13} {short(names[q], 48)} | {short(addrs[q], 70)}   {{{tail}}}"


rows = []
os.makedirs(OUT, exist_ok=True)
for c in NS:
    samp_c = [s for cc, s in sample if cc == c]
    qi = np.array([i for i in range(rec.height) if srcs[i] > 1 and ctry[i] == c])
    s1i = np.array([i for i in range(rec.height) if srcs[i] == 1 and ctry[i] == c])
    qn = [norm(names[i]) for i in qi]
    qa = [addr_w(addrs[i]) for i in qi]
    qe = np.array([not addrs[i].strip() for i in qi])
    s1n = [norm(names[i]) for i in s1i]
    s1a = [addr_w(addrs[i]) for i in s1i]
    sn = [norm(names[s]) for s in samp_c]
    sa = [addr_w(addrs[s]) for s in samp_c]
    t1 = time.time()
    NS_ = process.cdist(sn, qn, scorer=fuzz.token_set_ratio, dtype=np.uint8, workers=-1)
    AS_ = process.cdist(sa, qa, scorer=fuzz.token_set_ratio, dtype=np.uint8, workers=-1)
    NS1 = process.cdist(sn, s1n, scorer=fuzz.token_set_ratio, dtype=np.uint8, workers=-1)
    AS1 = process.cdist(sa, s1a, scorer=fuzz.token_set_ratio, dtype=np.uint8, workers=-1)
    print(c, "cdist", time.time() - t1, flush=True)
    with open(f"{OUT}/neigh_{c}.txt", "w", encoding="utf-8") as f:
        for k, s in enumerate(samp_c):
            mt = s1_matches.get(s, [])
            cd = [eid2i[x] for x in cand_of.get(eids[s], "").split(",") if x]
            ns, as_ = NS_[k].astype(np.int32), AS_[k].astype(np.int32)
            ok = ((as_ >= 90) & (ns >= 45)) | ((ns >= 80) & ((as_ >= 50) | qe))
            idx = np.nonzero(ok)[0]
            idx = idx[np.argsort(-(ns[idx] + as_[idx]), kind="stable")]
            seen = set(mt) | set(cd)
            extra = [int(qi[j]) for j in idx if int(qi[j]) not in seen][:10]
            ns1, as1 = NS1[k].astype(np.int32), AS1[k].astype(np.int32)
            ok1 = ((ns1 >= 80) & (as1 >= 60)) | ((as1 >= 95) & (ns1 >= 50))
            j1 = np.nonzero(ok1)[0]
            j1 = j1[np.argsort(-(ns1[j1] + as1[j1]), kind="stable")]
            others = [int(s1i[j]) for j in j1 if int(s1i[j]) != s][:5]
            f.write(f"=== #{k} {c} {eids[s]} nmatch={len(mt)} ncand={len(cd)}\n")
            f.write(f" S1 {short(names[s], 60)} | {short(addrs[s], 90)}\n")
            for q in mt:
                f.write(line("M", q, s) + "\n")
                rows.append((c, k, eids[s], "M", eids[q]))
            for q in cd:
                if q in mt:
                    continue
                f.write(line("c", q, s) + "\n")
                rows.append((c, k, eids[s], "c", eids[q]))
            for q in extra:
                f.write(line("n", q, s) + "\n")
                rows.append((c, k, eids[s], "n", eids[q]))
            for o in others:
                f.write(f" S {eids[o]:<13} {short(names[o], 48)} | {short(addrs[o], 70)}  {{nmatch={len(s1_matches.get(o, []))}}}\n")
                rows.append((c, k, eids[s], "S", eids[o]))
    del NS_, AS_
pl.DataFrame(rows, schema=["country", "k", "s1", "tag", "rid"], orient="row").write_parquet(f"{OUT}/neigh.parquet")
print("done", time.time() - t0, flush=True)

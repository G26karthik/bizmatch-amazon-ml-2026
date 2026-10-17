"""Blind audit B: build compact neighbourhoods for a random sample of S1 entities.

Output: /work/agent_blindB/neigh_<country>.txt (human-readable) + neigh.parquet (structured rows).
Tags: M = matched to this S1 in the submission; c = candidate (scores) not matched to this S1;
n = neighbour found by name/address similarity (not a candidate); S = other S1 nearby.
"""
import os
import sys
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

OUT = "/work/agent_blindB"
SUB = sys.argv[1] if len(sys.argv) > 1 else "/work/out_fr14/matching_results.tsv"
SEED = 20260927
NS = {"France": 250, "US": 100, "India": 60}
t0 = time.time()

rec = pl.read_parquet(
    "/work/runfr/test/records.parquet",
    columns=["entity_id", "business_name", "business_address", "country", "src", "name_n", "addr_n", "house", "addr_w", "addr_empty"],
).with_row_index("i")
eid2i = dict(zip(rec["entity_id"].to_list(), rec["i"].to_list()))
names = rec["business_name"].to_list()
addrs = rec["business_address"].to_list()
eids = rec["entity_id"].to_list()
houses = rec["house"].to_list()
print("records", rec.height, time.time() - t0, flush=True)

mr = pl.read_csv(SUB, separator="\t", schema_overrides={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8})
mr = mr.with_columns(pl.col("matched_entity_ids").fill_null(""))
assigned = {}  # q row -> s1 row
s1_matches = {}
for s, m in zip(mr["source1_entity_id"].to_list(), mr["matched_entity_ids"].to_list()):
    si = eid2i[s]
    lst = [eid2i[x] for x in m.split(",") if x]
    s1_matches[si] = lst
    for q in lst:
        assigned[q] = si
print("submission", mr.height, len(assigned), time.time() - t0, flush=True)

best = pl.read_parquet("/work/run28/test_best_run28_7q.parquet")
best_s1 = dict(zip(best["q"].to_list(), best["s1"].to_list()))
best_p = dict(zip(best["q"].to_list(), best["prob"].to_list()))
scores = pl.read_parquet("/work/run28/scores_test.parquet")

s1tab = rec.filter(pl.col("src") == 1).select("i", "country")
rng = np.random.default_rng(SEED)
sample = []
for c, n in NS.items():
    ids = np.sort(s1tab.filter(pl.col("country") == c)["i"].to_numpy())
    sample += [(c, int(x)) for x in rng.choice(ids, n, replace=False)]
samp_ids = [s for _, s in sample]
sc = scores.filter(pl.col("s1").is_in(samp_ids))
cand = {}
for q, s, p in zip(sc["q"].to_list(), sc["s1"].to_list(), sc["p3"].to_list()):
    cand.setdefault(s, {})[q] = p
print("sample+cands", len(sample), sc.height, time.time() - t0, flush=True)


def short(x, n):
    x = (x or "").replace("\t", " ")
    return x if len(x) <= n else x[: n - 1] + "~"


def line(tag, q, this):
    a = assigned.get(q)
    bs = best_s1.get(q)
    bp = best_p.get(q)
    p3 = cand.get(this, {}).get(q)
    parts = [f" {tag} {eids[q]:<13} {short(names[q], 48)} | {short(addrs[q], 70)}"]
    info = []
    if p3 is not None:
        info.append(f"p3={p3:.3f}")
    if bs is not None:
        info.append(f"best={'THIS' if bs == this else eids[bs]} p={bp:.4f}")
    if a is not None and a != this:
        info.append(f"-> {eids[a]} [{short(names[a], 36)} | {short(addrs[a], 50)}]")
    elif a is None and tag != "S":
        info.append("-> none")
    return parts[0] + "   {" + ", ".join(info) + "}"


rows = []
os.makedirs(OUT, exist_ok=True)
for c in NS:
    samp_c = [s for cc, s in sample if cc == c]
    sub = rec.filter((pl.col("country") == c) & (pl.col("src") > 1))
    s1c = rec.filter((pl.col("country") == c) & (pl.col("src") == 1))
    qi = sub["i"].to_numpy()
    s1i = s1c["i"].to_numpy()
    qn = [x or "" for x in sub["name_n"].to_list()]
    qa = [x or "" for x in sub["addr_w"].to_list()]
    s1n = [x or "" for x in s1c["name_n"].to_list()]
    s1a = [x or "" for x in s1c["addr_w"].to_list()]
    qe = sub["addr_empty"].to_numpy()
    sn = [rec["name_n"][s] or "" for s in samp_c]
    sa = [rec["addr_w"][s] or "" for s in samp_c]
    t1 = time.time()
    NS_ = process.cdist(sn, qn, scorer=fuzz.token_set_ratio, dtype=np.uint8, workers=2)
    AS_ = process.cdist(sa, qa, scorer=fuzz.token_set_ratio, dtype=np.uint8, workers=2)
    NS1 = process.cdist(sn, s1n, scorer=fuzz.token_set_ratio, dtype=np.uint8, workers=2)
    AS1 = process.cdist(sa, s1a, scorer=fuzz.token_set_ratio, dtype=np.uint8, workers=2)
    print(c, "cdist", time.time() - t1, flush=True)
    with open(f"{OUT}/neigh_{c}.txt", "w") as f:
        for k, s in enumerate(samp_c):
            mt = s1_matches.get(s, [])
            cd = cand.get(s, {})
            ns, as_ = NS_[k].astype(np.int32), AS_[k].astype(np.int32)
            ok = ((as_ >= 90) & (ns >= 45)) | ((ns >= 80) & ((as_ >= 50) | qe))
            idx = np.nonzero(ok)[0]
            idx = idx[np.argsort(-(ns[idx] + as_[idx]))]
            seen = set(mt) | set(cd)
            extra = [int(qi[j]) for j in idx if int(qi[j]) not in seen][:10]
            ns1, as1 = NS1[k].astype(np.int32), AS1[k].astype(np.int32)
            ok1 = ((ns1 >= 80) & (as1 >= 60)) | ((as1 >= 95) & (ns1 >= 50))
            j1 = np.nonzero(ok1)[0]
            j1 = j1[np.argsort(-(ns1[j1] + as1[j1]))]
            others = [int(s1i[j]) for j in j1 if int(s1i[j]) != s][:5]
            f.write(f"=== #{k} {c} {eids[s]} h={houses[s]} nmatch={len(mt)} ncand={len(cd)}\n")
            f.write(f" S1 {short(names[s], 60)} | {short(addrs[s], 90)}\n")
            for q in sorted(mt, key=lambda q: -cd.get(q, 0)):
                f.write(line("M", q, s) + "\n")
                rows.append((c, k, eids[s], "M", eids[q]))
            for q in sorted(cd, key=lambda q: -cd[q]):
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

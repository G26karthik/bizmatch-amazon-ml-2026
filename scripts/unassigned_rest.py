"""What are the unassigned records that no known decoy type explains? Per country: unassigned non-S1 records minus
(a) decoy-word records, (b) records whose name key equals / one-swaps an S1 name key on the same street at house + decoy
offset. The rest is classified by the best-matching S1 (same street, same name key elsewhere, nothing) and sampled.
A large France remainder resembling copies = missed copies (FN).

python scripts/unassigned_rest.py /work/out_fr7
"""
import random
import re
import sys
import unicodedata
from collections import Counter

import polars as pl

out = sys.argv[1]
LEGAL = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "inc", "llc", "ltd", "pvt", "private", "limited", "corp", "corporation",
         "incorporated", "company", "co", "lp", "llp", "pllc", "pc", "plc", "the", "de", "des", "du", "la", "le", "les", "et", "of", "and", "s", "a", "d", "l"}
DECOYW = {"US": {"southside", "eastgate", "northside", "midtown", "greater"},
          "France": {"holding", "participations", "distribution", "international", "groupe", "developpement", "snc"},
          "India": set()}
OFFS = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country", "business_name", "business_address", "addr_n", "house"]) \
        .with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
eid = rec.select("rid", "entity_id")
g = pl.read_csv(f"{out}/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = set(g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls("matched_entity_ids")
          .join(eid.rename({"entity_id": "matched_entity_ids"}), on="matched_entity_ids")["rid"].to_list())
names = rec["business_name"].to_list(); raw_addr = rec["business_address"].to_list(); addrs = rec["addr_n"].to_list(); houses = rec["house"].to_list()
src = rec["src"].to_list(); ctry = rec["country"].to_list()
nt = [set(re.findall(r"[a-z0-9]+", norm(n))) - LEGAL for n in names]
at = [set(re.findall(r"[a-z0-9]+", norm(a))) if a else set() for a in addrs]
street = [" ".join(sorted(t - {h})) if h else " ".join(sorted(t)) for t, h in zip(at, houses)]
hn = [int(h) if h and h.isdigit() else None for h in houses]
random.seed(5)
for country in ("US", "France"):
    idx = [i for i in range(len(names)) if ctry[i] == country]
    by_street, by_name = {}, {}
    for i in idx:
        if src[i] == 1:
            by_street.setdefault(street[i], []).append(i)
            by_name.setdefault(" ".join(sorted(nt[i])), []).append(i)
    n_s1 = sum(1 for i in idx if src[i] == 1)
    cls = Counter(); samples = {}
    for i in idx:
        if src[i] == 1 or i in acc:
            continue
        if nt[i] & DECOYW[country]:
            cls["decoy_word"] += 1
            continue
        best = None
        for s in by_street.get(street[i], []) if street[i] else []:
            A, B = nt[i], nt[s]
            rel_name = "same" if A == B else "swap1" if (len(A - B) <= 1 and len(B - A) <= 1 and A & B) else ("overlap" if A & B else "none")
            if hn[i] is not None and hn[s] is not None:
                dh = hn[i] - hn[s]
                rel_addr = "same" if dh == 0 and at[i] == at[s] else "offset" if dh in OFFS else "house_other"
            else:
                rel_addr = "nohouse"
            cand = (rel_name, rel_addr)
            rank = {"same": 0, "swap1": 1, "overlap": 2, "none": 3}
            if best is None or rank[cand[0]] < rank[best[0][0]]:
                best = (cand, s)
        if best is not None and best[0][1] == "offset" and best[0][0] in ("same", "swap1"):
            cls["offset_decoy"] += 1
            continue
        if best is not None and best[0][0] != "none":
            key = f"street:{best[0][0]}/{best[0][1]}"
        elif " ".join(sorted(nt[i])) in by_name:
            key = "name_elsewhere"
        elif not addrs[i]:
            key = "no_address"
        else:
            key = "nothing_similar"
        cls[key] += 1
        if len(samples.setdefault(key, [])) < 6 and random.random() < 0.05:
            samples[key].append((i, best[1] if best else None))
    tot = sum(cls.values())
    print(f"\n=== {country}: unassigned {tot} ({tot / n_s1:.2f}/S1)")
    for k, v in cls.most_common():
        print(f"  {k:32s} {v:8d}  {v / n_s1 * 1000:7.1f} per 1k S1")
    for k, ex in samples.items():
        print(f"  -- {k}")
        for i, s in ex:
            print(f"     REC {names[i]} | {raw_addr[i]}" + (f"\n     S1  {names[s]} | {raw_addr[s]}" if s is not None else ""))

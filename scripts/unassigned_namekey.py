"""Missed copies check: unassigned records whose name key (token set, legal/stop words removed) equals exactly ONE S1's name
key in the country, by address relation to that S1 and by the address-token overlap. US test F equals validation, so
US rates show what decoys/orphans look like; a France excess of 'same name, address changed' unassigned records would be
copies with address noise the model fails on (French address formats).

python scripts/unassigned_namekey.py /work/out_fr7
"""
import re
import sys
import unicodedata
from collections import Counter

import polars as pl

out = sys.argv[1]
LEGAL = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "inc", "llc", "ltd", "pvt", "private", "limited", "corp", "corporation",
         "incorporated", "company", "co", "lp", "llp", "pllc", "pc", "plc", "the", "de", "des", "du", "la", "le", "les", "et", "of", "and",
         "d", "l", "s", "a"}


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


def toks(s):
    return set(re.findall(r"[a-z0-9]+", norm(s)))


rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country", "business_name", "addr_n", "house"]).with_row_index("rid") \
        .with_columns(pl.col("rid").cast(pl.UInt32))
eid = rec.select("rid", "entity_id")
g = pl.read_csv(f"{out}/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls("matched_entity_ids") \
       .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
       .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1")
assigned = dict(zip(acc["q"].to_list(), acc["s1"].to_list()))
names = rec["business_name"].to_list(); addrs = rec["addr_n"].to_list(); houses = rec["house"].to_list()
src = rec["src"].to_list(); ctry = rec["country"].to_list()
examples = []
for country in ("US", "India", "France"):
    idx = [i for i in range(len(names)) if ctry[i] == country]
    key_s1 = {}
    for i in idx:
        if src[i] == 1:
            k = " ".join(sorted(toks(names[i]) - LEGAL))
            if k:
                key_s1.setdefault(k, []).append(i)
    n_s1 = sum(1 for i in idx if src[i] == 1)
    st = {"unassigned": Counter(), "accepted_same_s1": Counter()}
    for i in idx:
        if src[i] == 1:
            continue
        k = " ".join(sorted(toks(names[i]) - LEGAL))
        c = key_s1.get(k)
        if not c or len(c) != 1:
            continue
        s = c[0]
        a = assigned.get(i)
        grp = "unassigned" if a is None else ("accepted_same_s1" if a == s else None)
        if grp is None:
            continue
        if not addrs[i] or not addrs[s]:
            r = "missing"
        else:
            tq, ts = toks(addrs[i]), toks(addrs[s])
            if tq == ts:
                r = "same"
            else:
                wq, ws = {t for t in tq if not t.isdigit()}, {t for t in ts if not t.isdigit()}
                j = len(wq & ws) / max(len(wq | ws), 1)
                same_house = houses[i] and houses[i] == houses[s]
                r = ("house_only" if wq == ws else "same_house_street_edit" if same_house and j >= 0.5 else
                     "diff_house_street_similar" if j >= 0.5 else "same_house_street_diff" if same_house else "address_different")
        st[grp][r] += 1
        if country == "France" and grp == "unassigned" and r in ("same_house_street_edit", "diff_house_street_similar") and len(examples) < 25:
            examples.append((names[i], addrs[i], names[s], addrs[s]))
    for grp, c in st.items():
        print(f"{country} {grp} unique-name-key records per 1k S1: total {sum(c.values()) / n_s1 * 1000:.1f} | " +
              ", ".join(f"{k} {v / n_s1 * 1000:.1f}" for k, v in c.most_common()), flush=True)
print("\nFrance unassigned, same name key, address similar but changed:")
for e in examples:
    print("  REC", e[0], "|", e[1], "\n  S1 ", e[2], "|", e[3])

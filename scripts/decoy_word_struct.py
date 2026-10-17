"""Decoy-word structure per country. Records whose name carries a decoy word (US: southside, eastgate, ...; France:
holding, participations, ...) are generator decoys (US: never a true copy; 0 accepted). For each, strip the word, find the
S1 whose name key equals the rest, and report the address relation (same address, house + offset, other) -> tells how the
generator builds a decoy in each country. Then counts France accepted pairs whose record has a candidate decoy word
(groupe, developpement, ...) missing from the S1 name, by address relation, i.e. decoys our model accepted.

python scripts/decoy_word_struct.py /work/out_fr7
"""
import re
import sys
import unicodedata
from collections import Counter

import polars as pl

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_fmt_str_lengths(60); pl.Config.set_tbl_width_chars(250)
out = sys.argv[1]
LEGAL = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "inc", "llc", "ltd", "pvt", "private", "limited", "corp", "corporation",
         "incorporated", "company", "co", "lp", "llp", "pllc", "pc", "plc", "the", "de", "des", "du", "la", "le", "les", "et", "of", "and",
         "d", "l", "s", "a"}
WORDS = {"US": ["southside", "eastgate", "northside", "midtown", "greater"],
         "France": ["holding", "participations", "distribution", "international", "groupe", "developpement", "snc", "enterprises"]}


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


def akey(i):
    return " ".join(sorted(toks(addrs[i]) - {houses[i] or ""}))


def rel(q, s):
    if not addrs[q] or not addrs[s]:
        return "missing"
    if toks(addrs[q]) == toks(addrs[s]):
        return "same"
    if akey(q) == akey(s) and houses[q] and houses[s] and houses[q].isdigit() and houses[s].isdigit():
        d = int(houses[q]) - int(houses[s])
        return f"house{d:+d}" if d in (1, 2, 3, 4, 5, 7, 9, 11, 13, 21) else "house_other"
    return "other"


for country, words in WORDS.items():
    idx = [i for i in range(len(names)) if ctry[i] == country]
    key_s1 = {}
    for i in idx:
        if src[i] == 1:
            key_s1.setdefault(" ".join(sorted(toks(names[i]) - LEGAL)), []).append(i)
    for w in words:
        stats = Counter(); acc_stats = Counter(); n = 0
        for i in idx:
            if src[i] == 1:
                continue
            t = toks(names[i])
            if w not in t:
                continue
            n += 1
            k = " ".join(sorted(t - LEGAL - {w}))
            cands = key_s1.get(k, [])
            if not cands:
                stats["no_exact_s1"] += 1
                continue
            r = sorted(rel(i, s) for s in cands)[0] if len(cands) > 1 else rel(i, cands[0])
            stats[r if not r.startswith("house+") else "house+offset"] += 1
            if i in assigned:
                acc_stats["accepted_to_that_s1" if assigned[i] in cands else "accepted_elsewhere"] += 1
        print(f"{country} '{w}': {n} non-S1 records; S1 found by stripping the word: {n - stats['no_exact_s1']}; "
              f"address relation {dict(stats.most_common())}; {dict(acc_stats)}", flush=True)

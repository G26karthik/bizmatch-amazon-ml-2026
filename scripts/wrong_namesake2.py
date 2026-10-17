"""Wrong-namesake check with ORDER-INSENSITIVE addresses (France writes components in any order): for each accepted pair,
compare the record's address with the chosen S1's and with every same-name S1 (token-set name key, same country).
Levels: 3 = same address token set; 2 = same house + same street tokens; 1 = same street tokens; 0 = otherwise.
A same-name S1 at a higher level than the chosen S1 marks a likely wrong assignment.

python scripts/wrong_namesake2.py /work/out_fr7 [--write /work/dive/swap2.parquet]
"""
import re
import sys
import unicodedata

import polars as pl

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_fmt_str_lengths(55); pl.Config.set_tbl_width_chars(260)
out = sys.argv[1]
write = sys.argv[sys.argv.index("--write") + 1] if "--write" in sys.argv else None
STOP = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "de", "des", "du", "la", "le", "les", "et", "d", "l", "s", "a", "u", "r", "e",
        "inc", "llc", "ltd", "pvt", "private", "limited", "corp", "the", "of", "and", "co", "lp", "llp", "corporation", "incorporated", "company"}


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


def nkey(s):
    return " ".join(sorted(set(re.findall(r"[a-z0-9]+", norm(s))) - STOP))


def akeys(addr, house):
    toks = re.findall(r"[a-z0-9]+", norm(addr))
    words = frozenset(t for t in toks if not t.isdigit())
    return frozenset(toks), words


rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country", "business_name", "addr_n", "house"]).with_row_index("rid") \
        .with_columns(pl.col("rid").cast(pl.UInt32))
names = rec["business_name"].to_list(); addrs = rec["addr_n"].to_list(); houses = rec["house"].to_list()
NK = [nkey(n) for n in names]
AK = [akeys(a, h) for a, h in zip(addrs, houses)]


def level(i, j):
    if not addrs[i] or not addrs[j]:
        return 0
    if AK[i][0] == AK[j][0]:
        return 3
    if houses[i] and houses[i] == houses[j] and AK[i][1] == AK[j][1]:
        return 2
    if AK[i][1] == AK[j][1]:
        return 1
    return 0


eid = rec.select("rid", "entity_id")
g = pl.read_csv(f"{out}/matching_results.tsv", separator="\t")
p = g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls("matched_entity_ids") \
     .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
     .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1")
country = rec["country"].to_list()
src = rec["src"].to_list()
groups = {}
for i in range(len(names)):
    if src[i] == 1:
        groups.setdefault((country[i], NK[i]), []).append(i)
rows = []
for q, s in zip(p["q"].to_list(), p["s1"].to_list()):
    c = country[s]
    lv = level(q, s)
    best = (lv, s)
    for s2 in groups.get((c, NK[s]), ()):
        if s2 != s:
            l2 = level(q, s2)
            if l2 > best[0]:
                best = (l2, s2)
    rows.append((q, s, c, lv, best[0], best[1]))
d = pl.DataFrame(rows, schema=["q", "s1", "country", "lvl", "lvlb", "s1b"], orient="row")
print("accepted pairs: address level of chosen S1, by country (share):")
print(d.group_by("country", "lvl").len().with_columns((pl.col("len") / pl.col("len").sum().over("country")).round(4).alias("share")).sort("country", "lvl"))
sw = d.filter(pl.col("lvlb") > pl.col("lvl"))
print("pairs with a same-name S1 at a better address level (chosen -> better):")
print(sw.group_by("country", "lvl", "lvlb").len().sort("len", descending=True))
ex = sw.filter(pl.col("country") == "France").with_columns(pl.col("q").map_elements(lambda i: names[i] + " | " + addrs[i], return_dtype=pl.String).alias("record"),
                                                          pl.col("s1").map_elements(lambda i: addrs[i], return_dtype=pl.String).alias("chosen_addr"),
                                                          pl.col("s1b").map_elements(lambda i: addrs[i], return_dtype=pl.String).alias("better_addr"))
print(ex.sample(n=min(12, ex.height), seed=3).select("record", "chosen_addr", "better_addr", "lvl", "lvlb"))
if write:
    sw.write_parquet(write)

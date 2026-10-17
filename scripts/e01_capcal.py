"""E01 side check 4: n_other profile calibration on the fr8b accepted France set. For each accepted pair, n_other = other
fr8b-accepted records of the same S1 from the same source. Copies are bounded by the generator cap (S2 5, S3 6), so a copy
class has fewer records in slots near the cap than a decoy class. Compare certain copies (same address, identical /
case / punctuation / legal-only name), same-address non-word swaps and same-address word swaps (the fr10 target).

python scripts/e01_capcal.py
"""
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import polars as pl

sys.path.insert(0, "/work/agent_e01")
from e01_gap_lib import LEGALW, canon, name_rel  # noqa: E402

pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250)


def toks(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return [t for t in re.findall(r"[a-z0-9]+", s) if t not in LEGALW]


rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country", "business_name", "addr_n"]).with_row_index("rid") \
        .filter(pl.col("country") == "France")
eid = dict(zip(rec["entity_id"].to_list(), rec["rid"].to_list()))
nm = dict(zip(rec["rid"].to_list(), rec["business_name"].to_list()))
key = dict(zip(rec["rid"].to_list(), [" ".join(sorted(canon(a.split()))) for a in rec["addr_n"].to_list()]))
srcd = dict(zip(rec["rid"].to_list(), rec["src"].to_list()))
df = Counter(t for n in rec.filter(pl.col("src") == 1)["business_name"].to_list() for t in set(toks(n)))
rows = []
for line in open("/work/out_fr8b/matching_results.tsv"):
    a, _, b = line.rstrip("\n").partition("\t")
    if a in eid and b:
        rows.extend((eid[x], eid[a]) for x in b.split(",") if x in eid)
acc = pl.DataFrame(rows, schema={"q": pl.UInt32, "s1": pl.UInt32}, orient="row")
acc = acc.with_columns(pl.Series("src", [srcd[q] for q in acc["q"].to_list()], dtype=pl.Int8))
acc = acc.with_columns((pl.len().over("s1", "src") - 1).alias("n_other"))
f12 = set()
for line in open("/work/out_fr12/matching_results.tsv"):
    a, _, b = line.rstrip("\n").partition("\t")
    if a in eid and b:
        f12.update((eid[x], eid[a]) for x in b.split(",") if x in eid)
cls = []
for q, s in zip(acc["q"].to_list(), acc["s1"].to_list()):
    if not key[q] or key[q] != key[s]:
        cls.append("other_addr")
        continue
    r = name_rel(nm[q], nm[s]).split(":")[0]
    if r in ("exact", "case", "punct", "legal", "reorder"):
        cls.append("certain_copy")
        continue
    a_, b_ = Counter(toks(nm[q])), Counter(toks(nm[s]))
    i, o = list((a_ - b_).elements()), list((b_ - a_).elements())
    if len(i) == 1 and len(o) == 1:
        cls.append("swap_nonword" if df.get(i[0], 0) == 0 else "swap_word")
    else:
        cls.append("same_addr_other")
acc = acc.with_columns(pl.Series("cls", cls), pl.Series("in_fr12", [(q, s) in f12 for q, s in zip(acc["q"].to_list(), acc["s1"].to_list())]))
acc = acc.with_columns(pl.when(pl.col("cls").str.starts_with("swap") & ~pl.col("in_fr12")).then(pl.col("cls") + "_removed").otherwise(pl.col("cls")).alias("cls2"))
t = acc.group_by("cls2", "src", "n_other").len().with_columns((pl.col("len") / pl.col("len").sum().over("cls2", "src")).round(4).alias("share"))
w = t.pivot(on="n_other", index=["cls2", "src"], values="share", sort_columns=True).sort("src", "cls2")
print("=== fr8b accepted France pairs: n_other distribution (share within class) by class and source")
print(w)
print(acc.group_by("cls2", "src").agg(pl.len(), pl.col("n_other").mean().round(3).alias("mean_n_other")).sort("src", "cls2"))

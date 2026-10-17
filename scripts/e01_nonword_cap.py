"""E01 side check 2: cap test (model-free) on the fr8b -> fr12 France removals, split by the swapped-in word class.
Copies vanish where the S1's other accepted records in the record's source already fill the generator cap (S2 5, S3 6);
decoys do not. Rate per 1k S1-source slots by n_other (the S1's accepted records from that source in fr12).
Writes /work/agent_e01/fr_nonword_restore.parquet (q, s1, prob, reason): removed non-word swaps (candidate restorations).

python scripts/e01_nonword_cap.py
"""
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import polars as pl

sys.path.insert(0, "/work/agent_e01")
from e01_gap_lib import LEGALW  # noqa: E402

OUT = Path("/work/agent_e01")
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250)


def toks(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return [t for t in re.findall(r"[a-z0-9]+", s) if t not in LEGALW]


def swap(nq, ns):
    a, b = Counter(toks(nq)), Counter(toks(ns))
    i, o = list((a - b).elements()), list((b - a).elements())
    return (i[0], o[0]) if len(i) == 1 and len(o) == 1 else (None, None)


rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country", "business_name"]).with_row_index("rid")
fr = rec.filter(pl.col("country") == "France")
eid = dict(zip(fr["entity_id"].to_list(), fr["rid"].to_list()))
srcd = dict(zip(fr["rid"].to_list(), fr["src"].to_list()))


def load(p):
    rows = []
    for line in open(p):
        a, _, b = line.rstrip("\n").partition("\t")
        if a in eid and b:
            rows.extend((eid[x], eid[a]) for x in b.split(",") if x in eid)
    return pl.DataFrame(rows, schema={"q": pl.UInt32, "s1": pl.UInt32}, orient="row")


f8, f12 = load("/work/out_fr8b/matching_results.tsv"), load("/work/out_fr12/matching_results.tsv")
rm = f8.join(f12, on=["q", "s1"], how="anti")
df = Counter(t for n in fr.filter(pl.col("src") == 1)["business_name"].to_list() for t in set(toks(n)))
nm = dict(zip(fr["rid"].to_list(), fr["business_name"].to_list()))
cls = []
for q, s in rm.iter_rows():
    wi, wo = swap(nm[q], nm[s])
    if wi is None:
        cls.append("not_1swap")
    else:
        d = df.get(wi, 0)
        cls.append("nonword" if d == 0 else ("rare<20" if d < 20 else "word"))
rm = rm.with_columns(pl.Series("cls", cls), pl.Series("src", [srcd[q] for q in rm["q"].to_list()], dtype=pl.Int8))
acc = f12.with_columns(pl.Series("src", [srcd[q] for q in f12["q"].to_list()], dtype=pl.Int8))
cnt = {(s, sr): n for s, sr, n in acc.group_by("s1", "src").len().iter_rows()}
rm = rm.with_columns(pl.Series("n_other", [cnt.get((s, sr), 0) for s, sr in zip(rm["s1"].to_list(), rm["src"].to_list())]))
s1s = fr.filter(pl.col("src") == 1)["rid"].to_list()
slots = pl.DataFrame({"s1": s1s * 2, "src": [2] * len(s1s) + [3] * len(s1s)}).with_columns(pl.col("src").cast(pl.Int8))
slots = slots.with_columns(pl.Series("n_other", [cnt.get((s, sr), 0) for s, sr in zip(slots["s1"].to_list(), slots["src"].to_list())]))
den = slots.group_by("src", "n_other").agg(pl.len().alias("n_slots"))
t = rm.group_by("cls", "src", "n_other").agg(pl.len().alias("n")).join(den, on=["src", "n_other"])
full = ((pl.col("src") == 2) & (pl.col("n_other") >= 5)) | ((pl.col("src") == 3) & (pl.col("n_other") >= 6))
for nlo, nhi in ((1, 2), (2, 3), (3, 4)):
    lo = t.filter(pl.col("n_other").is_between(nlo, nhi)).group_by("cls").agg((pl.col("n").sum() / pl.col("n_slots").sum() * 1000).alias(f"rate_n{nlo}{nhi}"))
    if nlo == 1:
        z = lo
    else:
        z = z.join(lo, on="cls", how="full", coalesce=True)
cap = t.filter(full).group_by("cls").agg((pl.col("n").sum() / pl.col("n_slots").sum() * 1000).alias("rate_cap"), pl.col("n").sum().alias("n_at_cap"))
tot = rm.group_by("cls").agg(pl.len().alias("removed"))
z = tot.join(z, on="cls", how="left").join(cap, on="cls", how="left").with_columns((pl.col("rate_cap") / pl.col("rate_n23")).round(3).alias("cap_vs_n23"))
print("=== fr8b -> fr12 removed France pairs: cap test by swapped-in word class (rates per 1k S1-source slots)")
print(z.sort("removed", descending=True))
print("\nby source, non-word vs word:")
print(t.filter(pl.col("cls").is_in(["nonword", "word"])).with_columns((pl.col("n") / pl.col("n_slots") * 1000).round(3).alias("rate"))
       .sort("cls", "src", "n_other").select("cls", "src", "n_other", "n", "n_slots", "rate"))
b7 = pl.read_parquet("/work/run28/test_best_run28_7q.parquet")
out = rm.filter(pl.col("cls") == "nonword").join(b7, on=["q", "s1"], how="left") \
        .with_columns(pl.lit("fr10/fr12 removed same-address swap to a non-word (df 0 in France S1 names)").alias("reason"))
print("restore candidates:", out.height, "prob (7q, when best):", out["prob"].describe())
out.select("q", "s1", "prob", "reason").write_parquet(OUT / "fr_nonword_restore.parquet")

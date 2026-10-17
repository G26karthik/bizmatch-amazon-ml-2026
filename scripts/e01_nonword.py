"""E01 side check: the fr10 / fr12 France removals (same-address one-word type swaps) include swaps whose new word is not a
word at all ("Team Cmb SA" / "Team Club SA", "Bottom Asi" / "Bottom Amis", "Nantes Mnnson" / "Nantes Maison"): generator
name noise, while sibling decoys swap to a real vocabulary word. Counts removed pairs by whether the swapped-in word occurs
in any France S1 name (df), and checks the analogue on labelled US/India validation: same-address one-token swaps to a
non-word (df 0 among the country's S1 names) -> precision.

python scripts/e01_nonword.py
"""
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import polars as pl

sys.path.insert(0, "src")
sys.path.insert(0, "/work/agent_e01")
from e01_gap_lib import LEGALW  # noqa: E402

OUT = Path("/work/agent_e01")
pl.Config.set_fmt_str_lengths(50); pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(300)


def toks(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return [t for t in re.findall(r"[a-z0-9]+", s) if t not in LEGALW]


def swap(nq, ns):
    a, b = Counter(toks(nq)), Counter(toks(ns))
    i, o = list((a - b).elements()), list((b - a).elements())
    return (i[0], o[0]) if len(i) == 1 and len(o) == 1 else (None, None)


rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country", "business_name", "addr_n"]).with_row_index("rid")
fr = rec.filter(pl.col("country") == "France")
eid = dict(zip(fr["entity_id"].to_list(), fr["rid"].to_list()))


def load(p):
    rows = []
    for line in open(p):
        a, _, b = line.rstrip("\n").partition("\t")
        if a in eid and b:
            rows.extend((eid[x], eid[a]) for x in b.split(",") if x in eid)
    return pl.DataFrame(rows, schema={"q": pl.UInt32, "s1": pl.UInt32}, orient="row")


f8, f12 = load("/work/out_fr8b/matching_results.tsv"), load("/work/out_fr12/matching_results.tsv")
rm = f8.join(f12, on=["q", "s1"], how="anti")
ad = f12.join(f8, on=["q", "s1"], how="anti")
print(f"fr8b France {f8.height}, fr12 {f12.height}: removed {rm.height}, added {ad.height}")
df = Counter(t for n in fr.filter(pl.col("src") == 1)["business_name"].to_list() for t in set(toks(n)))
R = fr.select("rid", "business_name", "addr_n")
x = rm.join(R.rename({"rid": "q", "business_name": "nq", "addr_n": "aq"}), on="q").join(R.rename({"rid": "s1", "business_name": "ns", "addr_n": "as_"}), on="s1")
sw = [swap(a, b) for a, b in zip(x["nq"].to_list(), x["ns"].to_list())]
x = x.with_columns(pl.Series("w_in", [s[0] for s in sw]), pl.Series("w_out", [s[1] for s in sw]))
x = x.with_columns(pl.col("w_in").map_elements(lambda w: df.get(w, 0), return_dtype=pl.Int64).alias("df_in"),
                   (pl.col("w_in").str.slice(0, 1) == pl.col("w_out").str.slice(0, 1)).alias("same_first"))
x = x.with_columns(pl.when(pl.col("w_in").is_null()).then(pl.lit("not_1swap")).when(pl.col("df_in") == 0).then(pl.lit("nonword"))
                   .when(pl.col("df_in") < 20).then(pl.lit("rare<20")).otherwise(pl.lit("word")).alias("cls"))
print(x.group_by("cls", "same_first").agg(pl.len()).sort("len", descending=True))
print(x.filter(pl.col("cls") == "nonword").select("nq", "ns", "w_in", "w_out", "aq", "as_").sample(30, seed=1))
x.filter(pl.col("cls") == "nonword").select("q", "s1", "nq", "ns", "w_in", "w_out", "same_first").write_parquet(OUT / "fr12_removed_nonword.parquet")

# validation analogue (US / India): same-address one-token swap to a non-word
from ber.pipeline import labels  # noqa: E402
from ber.stage2 import _chosen  # noqa: E402

work = Path("/work/run28")
lab = labels(Path("/work/data"), work).select(pl.col("rid").cast(pl.UInt32).alias("q"), "parent")
vb = pl.read_parquet(work / "val_best_run28_7q.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
ch = _chosen(vb, {"mode": "efopt", "miss": 0.0, "gamma": 1.5}).select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
tr = pl.read_parquet(work / "train" / "records.parquet", columns=["src", "country", "business_name", "addr_n"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
dfc = {}
for c in ("US", "India"):
    dfc[c] = Counter(t for n in tr.filter((pl.col("src") == 1) & (pl.col("country") == c))["business_name"].to_list() for t in set(toks(n)))
v = vb.join(tr.select(pl.col("rid").alias("q"), "country", pl.col("business_name").alias("nq"), pl.col("addr_n").alias("aq")), on="q") \
      .join(tr.select(pl.col("rid").alias("s1"), pl.col("business_name").alias("ns"), pl.col("addr_n").alias("as_")), on="s1") \
      .join(lab, on="q", how="left").join(ch.with_columns(pl.lit(True).alias("acc")), on=["q", "s1"], how="left") \
      .with_columns(pl.col("acc").fill_null(False), (pl.col("parent").cast(pl.UInt32) == pl.col("s1")).fill_null(False).alias("y"))
v = v.filter((pl.col("aq") != "") & (pl.Series([" ".join(sorted(a.split())) == " ".join(sorted(b.split())) for a, b in zip(v["aq"].to_list(), v["as_"].to_list())])))
sw = [swap(a, b) for a, b in zip(v["nq"].to_list(), v["ns"].to_list())]
v = v.with_columns(pl.Series("w_in", [s[0] for s in sw]), pl.Series("w_out", [s[1] for s in sw])).filter(pl.col("w_in").is_not_null())
v = v.with_columns(pl.Series("df_in", [dfc.get(c, {}).get(w, 0) for c, w in zip(v["country"].to_list(), v["w_in"].to_list())]))
v = v.with_columns(pl.when(pl.col("df_in") == 0).then(pl.lit("nonword")).when(pl.col("df_in") < 20).then(pl.lit("rare<20")).otherwise(pl.lit("word")).alias("cls"),
                   (pl.col("w_in").str.slice(0, 1) == pl.col("w_out").str.slice(0, 1)).alias("same_first"))
print("\n=== validation: same-address one-token swaps (best S1), precision by swapped-in word class")
print(v.group_by("country", "cls", "same_first").agg(pl.len(), pl.col("y").mean().alias("prec"), pl.col("acc").mean().alias("acc_rate"),
                                                     pl.col("y").filter(~pl.col("acc")).mean().alias("prec_unacc")).sort("country", "cls", "same_first"))

"""E01 sanity check on labelled US/India validation: precision of the add rule "record's best S1 has the identical address
(addr_n token multiset) and a near-identical name (exact / case / punctuation / legal-form only / one-token typo)", split
by whether the production decision (run28_7q, efopt gamma 1.5) already accepts the pair; plus the same for addresses
that are equal only after mapping US abbreviations the normaliser leaves (a gap analogue: apt/unit/#, ste/suite ...).

python scripts/e01_val.py
"""
import sys
from collections import Counter
from pathlib import Path

import polars as pl

sys.path.insert(0, "src")
sys.path.insert(0, "/work/agent_e01")
from ber.pipeline import labels  # noqa: E402
from ber.stage2 import _chosen  # noqa: E402

from e01_gap_lib import name_rel, toks  # noqa: E402

pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(60)
work = Path("/work/run28")
lab = labels(Path("/work/data"), work)
parent = lab.select(pl.col("rid").alias("q").cast(pl.UInt32), "parent")
vb = pl.read_parquet(work / "val_best_run28_7q.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
ch = _chosen(vb, {"mode": "efopt", "miss": 0.0, "gamma": 1.5}).select(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
rec = pl.read_parquet(work / "train" / "records.parquet", columns=["src", "country", "business_name", "addr_n"]).with_row_index("rid") \
        .with_columns(pl.col("rid").cast(pl.UInt32))
d = vb.join(rec.rename({"rid": "q", "src": "src_q", "business_name": "nq", "addr_n": "aq", "country": "cty"}), on="q") \
      .join(rec.select(pl.col("rid").alias("s1"), pl.col("business_name").alias("ns"), pl.col("addr_n").alias("as_")), on="s1") \
      .join(parent, on="q", how="left") \
      .join(ch.with_columns(pl.lit(True).alias("acc")), on=["q", "s1"], how="left").with_columns(pl.col("acc").fill_null(False))
d = d.with_columns((pl.col("parent").cast(pl.UInt32) == pl.col("s1")).fill_null(False).alias("y"))
d = d.filter(pl.col("aq") != "")
kq = [" ".join(sorted(a.split())) for a in d["aq"].to_list()]
ks = [" ".join(sorted(a.split())) for a in d["as_"].to_list()]
US_GAP = {"unit": "apt", "#": "apt", "ste": "apt", "rm": "apt", "room": "apt", "fl": "fl", "flr": "fl", "bldg": "bldg", "bld": "bldg",
          "pkwy": "pkwy", "pky": "pkwy", "expy": "expy", "expressway": "expy", "fwy": "fwy", "freeway": "fwy", "ln": "ln", "pl": "pl",
          "rte": "rte", "route": "rte", "ctr": "ctr", "center": "ctr", "cntr": "ctr", "hts": "hts", "heights": "hts", "blvd": "blvd",
          "jn": "jn", "crossing": "xing", "xing": "xing", "rd": "rd", "po": "po", "box": "box", "sect": "sec", "sec": "sec",
          "mg": "mrg", "rd.": "rd", "opp": "opp", "nr": "nr", "ngr": "ngr", "ngar": "ngr", "colony": "col", "col": "col",
          "chowk": "chk", "chk": "chk", "bazar": "bzr", "bazaar": "bzr", "bzr": "bzr", "extn": "extn", "extension": "extn", "ext": "extn",
          "phase": "ph", "ph": "ph", "block": "blk", "blk": "blk", "plot": "plot", "flat": "flat", "house": "h", "h": "h", "no": "no"}


def canon(a):
    return " ".join(sorted(US_GAP.get(t, t) for t in a.split()))


kqa = [canon(a) for a in d["aq"].to_list()]
ksa = [canon(a) for a in d["as_"].to_list()]
d = d.with_columns(pl.Series("eq", [a == b for a, b in zip(kq, ks)]), pl.Series("eq_after", [a == b for a, b in zip(kqa, ksa)]))
rel = [name_rel(a, b) for a, b in zip(d["nq"].to_list(), d["ns"].to_list())]
d = d.with_columns(pl.Series("rel", [r.split(":")[0] for r in rel]))
STRONG = ["exact", "case", "punct", "legal", "typo", "reorder"]
print("=== val best pairs with identical address (token multiset), by country / name relation")
print(d.filter(pl.col("eq")).group_by("cty", "rel").agg(pl.len().alias("n"), pl.col("y").mean().alias("prec_all"), pl.col("acc").mean().alias("acc_rate"),
                                                         pl.col("y").filter(~pl.col("acc")).mean().alias("prec_unacc"),
                                                         (~pl.col("acc")).sum().alias("n_unacc")).sort("cty", "n", descending=[False, True]))
print("\n=== val best pairs equal only after US/India abbreviation mapping (gap analogue)")
g = d.filter(pl.col("eq_after") & ~pl.col("eq"))
print(g.group_by("cty", "rel").agg(pl.len().alias("n"), pl.col("y").mean().alias("prec_all"), pl.col("acc").mean().alias("acc_rate"),
                                   pl.col("y").filter(~pl.col("acc")).mean().alias("prec_unacc"), (~pl.col("acc")).sum().alias("n_unacc")).sort("cty", "n", descending=[False, True]))
print("\n=== unaccepted, identical address, strong name: prob distribution and precision by prob bin")
u = d.filter(pl.col("eq") & ~pl.col("acc") & pl.col("rel").is_in(STRONG))
print(u.with_columns(pl.col("prob").cut([0.05, 0.2, 0.5, 0.8]).alias("bin")).group_by("bin").agg(pl.len(), pl.col("y").mean()).sort("bin"))
print(u.select("nq", "ns", "aq", "as_", "prob", "y").head(12))

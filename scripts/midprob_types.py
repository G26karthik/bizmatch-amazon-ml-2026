"""What are the unaccepted best-S1 pairs with mid/high probability (0.2-0.95)? France has ~6x more per S1 than US.
Classifies by copy type (name op x address op) plus decoy flags (decoy word in record name, house = S1 house + decoy
offset), per country and probability band, with samples for France.

python scripts/midprob_types.py /work/out_run28_7q run28_7q
"""
import random
import re
import sys
import unicodedata

import polars as pl

_src = open("/work/code/scripts/op_mix.py").read().split("# --- train truth")[0]
_ns = {"__name__": "op_mix_helpers"}
_argv = sys.argv; sys.argv = sys.argv[:1] + ["-"]
exec(_src, _ns)
sys.argv = _argv
name_op, addr_op = _ns["name_op"], _ns["addr_op"]
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(250)
out, tag = sys.argv[1], sys.argv[2]
DW = {"southside", "eastgate", "northside", "midtown", "greater", "holding", "participations", "distribution", "international", "snc"}
OFFS = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


rec = pl.read_parquet("/work/run28/test/records.parquet", columns=["entity_id", "src", "country", "business_name", "business_address", "addr_n", "house"]) \
        .with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
eid = rec.select("rid", "entity_id")
tb = pl.read_parquet(f"/work/run28/test_best_{tag}.parquet")
g = pl.read_csv(f"{out}/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls().filter(pl.col("matched_entity_ids") != "") \
       .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
       .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1", pl.lit(True).alias("acc"))
d = tb.filter((pl.col("prob") > 0.2) & (pl.col("prob") <= 0.95)).join(acc, on=["q", "s1"], how="left").with_columns(pl.col("acc").fill_null(False))
d = d.filter(~pl.col("acc"))
cols = {"business_name": "n", "addr_n": "a", "house": "h", "business_address": "r"}
d = d.join(rec.select(pl.col("rid").alias("q"), *[pl.col(k).alias(v + "q") for k, v in cols.items()]), on="q") \
     .join(rec.select(pl.col("rid").alias("s1"), "country", *[pl.col(k).alias(v + "s") for k, v in cols.items()]), on="s1")
dw = [bool(set(re.findall(r"[a-z]+", norm(a))) & DW) and not (set(re.findall(r"[a-z]+", norm(b))) & DW) for a, b in zip(d["nq"].to_list(), d["ns"].to_list())]
off = []
for hq, hs in zip(d["hq"].to_list(), d["hs"].to_list()):
    off.append(bool(hq and hs and hq.isdigit() and hs.isdigit() and (int(hq) - int(hs)) in OFFS))
d = d.with_columns(pl.Series("decoyword", dw), pl.Series("offset", off),
                   pl.Series("name_op", [name_op(a, b) for a, b in zip(d["nq"].to_list(), d["ns"].to_list())]),
                   pl.Series("addr_op", [addr_op(*r) for r in zip(d["aq"].to_list(), d["hq"].to_list(), d["as"].to_list(), d["hs"].to_list())]))
d = d.with_columns(pl.when(pl.col("decoyword")).then(pl.lit("DECOYWORD")).when(pl.col("offset")).then(pl.lit("OFFSET"))
                   .otherwise(pl.concat_str([pl.col("name_op"), pl.col("addr_op")], separator="|")).alias("t"),
                   pl.when(pl.col("prob") <= 0.5).then(pl.lit("p0.2-0.5")).otherwise(pl.lit("p0.5-0.95")).alias("band"))
n_s1 = rec.filter(pl.col("src") == 1).group_by("country").len().rename({"len": "n_s1"})
t = d.group_by("country", "band", "t").len().join(n_s1, on="country").with_columns((pl.col("len") / pl.col("n_s1") * 1000).round(2).alias("per1k"))
for band in ("p0.5-0.95", "p0.2-0.5"):
    x = t.filter(pl.col("band") == band).pivot(on="country", index="t", values="per1k").fill_null(0).sort("France", descending=True)
    print(f"\n=== unaccepted best pairs, {band}, per 1k S1")
    print(x.head(22))
random.seed(1)
fr = d.filter((pl.col("country") == "France") & (pl.col("band") == "p0.5-0.95") & ~pl.col("t").is_in(["DECOYWORD", "OFFSET"]))
for row in fr.sample(n=30, seed=3).iter_rows(named=True):
    print(f"  p={row['prob']:.2f} [{row['t']}]\n     REC {row['nq']} | {row['rq']}\n     S1  {row['ns']} | {row['rs']}")

"""Singleton-with-decoy test. Copies of an S1 draw their noise type independently, so the copy-type mix of the records of
S1s with exactly ONE predicted copy should equal the mix among S1s with 3+ predicted copies. An excess of some type among
k=1 S1s = true singletons (or wrong S1s) that got a decoy of that type. US (test F = validation) calibrates the check.

python scripts/k1_mix.py /work/out_fr7
"""
import sys

import polars as pl

_src = open("/work/code/scripts/op_mix.py").read().split("# --- train truth")[0]
_ns = {"__name__": "op_mix_helpers"}
_argv = sys.argv; sys.argv = sys.argv[:1] + ["-"]
exec(_src, _ns)
sys.argv = _argv
name_op, addr_op = _ns["name_op"], _ns["addr_op"]
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(250)
out = sys.argv[1]
rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country", "business_name", "addr_n", "house"]).with_row_index("rid") \
        .with_columns(pl.col("rid").cast(pl.UInt32))
eid = rec.select("rid", "entity_id")
g = pl.read_csv(f"{out}/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
p = g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls("matched_entity_ids") \
     .filter(pl.col("matched_entity_ids") != "") \
     .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
     .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1")
p = p.join(p.group_by("s1").len().rename({"len": "k"}), on="s1")
p = p.with_columns(pl.when(pl.col("k") == 1).then(pl.lit("k1")).when(pl.col("k") == 2).then(pl.lit("k2")).otherwise(pl.lit("k3+")).alias("kb"))
cols = {"business_name": "n", "addr_n": "a", "house": "h"}
p = p.join(rec.select(pl.col("rid").alias("q"), *[pl.col(k).alias(v + "q") for k, v in cols.items()]), on="q") \
     .join(rec.select(pl.col("rid").alias("s1"), "country", *[pl.col(k).alias(v + "s") for k, v in cols.items()]), on="s1")
# k3+ is large: sample to keep the classification fast
p = pl.concat([p.filter(pl.col("kb") != "k3+"), p.filter(pl.col("kb") == "k3+").sample(fraction=0.25, seed=1)])
p = p.with_columns(pl.Series("name_op", [name_op(a, b) for a, b in zip(p["nq"].to_list(), p["ns"].to_list())]),
                   pl.Series("addr_op", [addr_op(*r) for r in zip(p["aq"].to_list(), p["hq"].to_list(), p["as"].to_list(), p["hs"].to_list())]))
p = p.with_columns(pl.concat_str([pl.col("name_op"), pl.col("addr_op")], separator="|").alias("t"))
sh = p.group_by("country", "kb", "t").len().with_columns((pl.col("len") / pl.col("len").sum().over("country", "kb")).alias("share"))
for c in ("US", "India", "France"):
    x = sh.filter(pl.col("country") == c).pivot(on="kb", index="t", values="share").fill_null(0)
    n1 = p.filter((pl.col("country") == c) & (pl.col("kb") == "k1")).height
    x = x.with_columns(((pl.col("k1") - pl.col("k3+")) * n1).round(0).alias("excess_k1_records"), (pl.col("k1") / pl.col("k3+")).round(2).alias("ratio"))
    print(f"\n=== {c}: k1 records {n1}; types by excess among k=1 S1s vs k>=3")
    print(x.sort("excess_k1_records", descending=True).head(14))

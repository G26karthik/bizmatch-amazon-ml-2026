"""France decisions of two submissions: pairs only in A, only in B, by copy type (name change x address change), with
examples. Used to see what newer models changed in France (v26 France ~0.9547 vs run28-based ~0.952-0.954 on the LB).

python scripts/fr_version_diff.py <A matching_results.tsv> <B matching_results.tsv>
"""
import sys

import polars as pl

# op_mix runs its report at import; take only its helper definitions
_src = open("/work/code/scripts/op_mix.py").read().split("# --- train truth")[0]
_ns = {"__name__": "op_mix_helpers"}
sys.argv, _argv = sys.argv[:1] + ["-"], sys.argv
exec(_src, _ns)
sys.argv = _argv
name_op, addr_op = _ns["name_op"], _ns["addr_op"]

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_fmt_str_lengths(55); pl.Config.set_tbl_width_chars(250)
A, B = sys.argv[1], sys.argv[2]
rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country", "business_name", "addr_n", "house"]).with_row_index("rid") \
        .with_columns(pl.col("rid").cast(pl.UInt32))
eid = rec.select("rid", "entity_id")
fr = rec.filter((pl.col("country") == "France") & (pl.col("src") == 1)).select(pl.col("rid").alias("s1"))


def load(path):
    g = pl.read_csv(path, separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
    return g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls("matched_entity_ids") \
            .filter(pl.col("matched_entity_ids") != "") \
            .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
            .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1").join(fr, on="s1", how="semi")


a, b = load(A), load(B)
print(f"France pairs: A {a.height}, B {b.height}, common {a.join(b, on=['q', 's1']).height}")
for name, x in (("only in A", a.join(b, on=["q", "s1"], how="anti")), ("only in B", b.join(a, on=["q", "s1"], how="anti"))):
    x = x.join(rec.select(pl.col("rid").alias("q"), pl.col("business_name").alias("nq"), pl.col("addr_n").alias("aq"), pl.col("house").alias("hq")), on="q") \
         .join(rec.select(pl.col("rid").alias("s1"), pl.col("business_name").alias("ns"), pl.col("addr_n").alias("as"), pl.col("house").alias("hs")), on="s1")
    x = x.with_columns(pl.Series("name_op", [name_op(p, q) for p, q in zip(x["nq"].to_list(), x["ns"].to_list())]),
                       pl.Series("addr_op", [addr_op(*r) for r in zip(x["aq"].to_list(), x["hq"].to_list(), x["as"].to_list(), x["hs"].to_list())]))
    # does the other file assign this record elsewhere?
    other = b if name == "only in A" else a
    x = x.join(other.rename({"s1": "s1_other"}), on="q", how="left")
    print(f"\n=== {name}: {x.height} pairs; record assigned to another S1 in the other file: {x['s1_other'].is_not_null().sum()}")
    print(x.group_by("name_op", "addr_op").len().sort("len", descending=True).head(15))
    print(x.sample(n=min(10, x.height), seed=7).select("nq", "ns", "aq", "as"))

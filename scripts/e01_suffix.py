"""E01 step 3: the suffix-collapse side. House parsing keeps only the digits ("12 bis" -> house 12), so 12 / 12 bis /
12 ter share house_eq. Pairs whose addresses (canonical gap tokens) differ ONLY by a house suffix (bis/ter/quater present
on one side, absent or different on the other): counts, acceptance and name relation vs the same-address control, and
whether a competing S1 at the exact address exists. Also: how often plausible copies (exact-name pairs) of S1s with a
suffix drop it.

python scripts/e01_suffix.py   (after scripts/e01_gap.py)
"""
import sys
from collections import Counter
from pathlib import Path

import polars as pl

sys.path.insert(0, "/work/agent_e01")
from e01_gap_lib import DECOY, canon, name_rel, toks  # noqa: E402

OUT = Path("/work/agent_e01")
pl.Config.set_fmt_str_lengths(60); pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(300)
SUF = {"bis", "ter", "quater"}

rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country", "business_name", "addr_n", "house"]) \
        .with_row_index("rid").filter(pl.col("country") == "France").drop("country")
tl = [a.split() for a in rec["addr_n"].to_list()]
ca = [canon(t) for t in tl]
# canonical key without house suffixes
key_ns = [" ".join(sorted(x for x in t if x not in SUF)) for t in ca]
key_a = [" ".join(sorted(t)) for t in ca]
suf = [",".join(sorted(x for x in t if x in SUF)) for t in ca]
rec = rec.with_columns(pl.Series("key_ns", key_ns), pl.Series("key_a", key_a), pl.Series("suf", suf))
eid = dict(zip(rec["entity_id"].to_list(), rec["rid"].to_list()))


def load_tsv(p):
    rows = []
    for line in open(p):
        a, _, b = line.rstrip("\n").partition("\t")
        if a not in eid or not b:
            continue
        rows.extend((eid[x], eid[a]) for x in b.split(",") if x in eid)
    return pl.DataFrame(rows, schema={"q": pl.UInt32, "s1": pl.UInt32}, orient="row")


acc = load_tsv("/work/out_fr12/matching_results.tsv")
p3 = pl.read_parquet("/work/runfr28/scores_test.parquet")
best7 = pl.read_parquet("/work/run28/test_best_run28_7q.parquet").rename({"s1": "best7", "prob": "p7"})
cand = pl.concat([pl.read_parquet(p, columns=["q", "s1", "street_tsr", "house_eq"]) for p in sorted(Path("/work/runfr/test/pruned").glob("France_*.parquet"))]) \
         .with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
R = rec.select("rid", "src", "business_name", "addr_n", "key_ns", "key_a", "suf")
c = cand.join(R.rename({x: x + "_q" for x in R.columns if x != "rid"}), left_on="q", right_on="rid") \
        .join(R.rename({x: x + "_s" for x in R.columns if x != "rid"}), left_on="s1", right_on="rid") \
        .filter(pl.col("addr_n_q") != "")
c = c.filter(pl.col("key_ns_q") == pl.col("key_ns_s")).with_columns(
    pl.when(pl.col("suf_q") == pl.col("suf_s")).then(pl.lit("same"))
      .when(pl.col("suf_q") == "").then(pl.lit("s1_suffix_dropped"))
      .when(pl.col("suf_s") == "").then(pl.lit("q_suffix_added"))
      .otherwise(pl.lit("suffix_changed")).alias("kind"))
c = c.join(acc.with_columns(pl.lit(True).alias("acc")), on=["q", "s1"], how="left").with_columns(pl.col("acc").fill_null(False))
asg = acc.rename({"s1": "asg_s1"}).group_by("q").agg(pl.col("asg_s1").first())
c = c.join(asg, on="q", how="left").with_columns(
    pl.when(pl.col("acc")).then(pl.lit("this")).when(pl.col("asg_s1").is_null()).then(pl.lit("unassigned")).otherwise(pl.lit("other")).alias("status"))
c = c.join(p3, on=["q", "s1"], how="left").join(best7, on="q", how="left")
rel = [name_rel(a, b).split(":")[0] for a, b in zip(c["business_name_q"].to_list(), c["business_name_s"].to_list())]
c = c.with_columns(pl.Series("rel", rel))
c = c.with_columns(pl.col("rel").is_in(["exact", "case", "punct", "legal", "reorder"]).alias("strong"))
print("=== France candidate pairs, same address apart from the house suffix (canonical gap tokens)")
print(c.group_by("kind", "suf_s", "suf_q").agg(pl.len(), pl.col("acc").mean().alias("acc"), pl.col("strong").mean().alias("strong_name"),
                                           pl.col("acc").filter(pl.col("strong")).mean().alias("acc_strong"),
                                           pl.col("acc").filter(~pl.col("strong")).mean().alias("acc_weak"),
                                           pl.col("p3").median()).sort("len", descending=True).head(20))
print(c.group_by("kind", "rel").agg(pl.len(), pl.col("acc").mean()).sort("kind", "len", descending=[False, True]).head(40))

# competing S1 at the exact (suffix-equal) address for records accepted across a suffix mismatch
mm = c.filter((pl.col("kind") != "same") & pl.col("acc"))
exact = c.filter(pl.col("kind") == "same").select("q", pl.col("s1").alias("s1_exact"), pl.col("rel").alias("rel_exact"), pl.col("p3").alias("p3_exact"))
mm = mm.join(exact, on="q", how="left")
print(f"\naccepted suffix-mismatch pairs {mm['q'].n_unique()} records; with a candidate S1 at the exact address: "
      f"{mm.filter(pl.col('s1_exact').is_not_null())['q'].n_unique()}")
print(mm.filter(pl.col("s1_exact").is_not_null()).select("q", "s1", "s1_exact", "business_name_q", "business_name_s", "rel", "rel_exact",
                                                         "addr_n_q", "addr_n_s", "p3", "p3_exact").head(30))
print(mm.filter(pl.col("s1_exact").is_not_null()).group_by("rel", "rel_exact").len().sort("len", descending=True).head(20))

# generator: S1s with a suffix -> how do their strong-name accepted records write it (kept / dropped / changed)?
s1suf = c.filter((pl.col("suf_s") != "") & pl.col("strong") & pl.col("acc"))
print("\nstrong-name accepted records of S1s with a suffix:", s1suf.group_by("kind").len())
s1no = c.filter((pl.col("suf_s") == "") & pl.col("strong") & pl.col("acc"))
print("strong-name accepted records of S1s without suffix:", s1no.group_by("kind").len())
# weak-name (type swaps etc.) suffix mismatches: model acceptance
c.filter(pl.col("kind") != "same").write_parquet(OUT / "suffix_pairs.parquet")

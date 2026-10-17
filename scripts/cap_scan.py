"""Cap scan over all record categories. For every record whose best S1 is s (test_best, accepted or not), category =
name op x address op (op_mix classes). n_other = accepted records of s in the record's source, excluding the record.
Copies cannot exist once n_other reaches the cap (S2: 5, S3: 6), decoys can. Per category: rate per 1k S1 at each n_other
and the at-cap rate relative to n_other = 3, against a pure-copy reference (identical|same). High at-cap presence =
the category holds decoys. US is the calibration (its accepted set is validated).

python scripts/cap_scan.py /work/out_fr10
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
CAP = {2: 5, 3: 6}
rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country", "business_name", "addr_n", "house"]).with_row_index("rid") \
        .with_columns(pl.col("rid").cast(pl.UInt32))
eid = rec.select("rid", "entity_id")
g = pl.read_csv(f"{out}/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls().filter(pl.col("matched_entity_ids") != "") \
       .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
       .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1", pl.lit(True).alias("acc"))
tb = pl.read_parquet("/work/run28/test_best_run28_7q.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32)).select("q", "s1")
# pool: best-S1 pairs plus accepted pairs (rule additions may differ from the best S1)
pool = pl.concat([tb, acc.select("q", "s1")]).unique()
pool = pool.join(acc, on=["q", "s1"], how="left").with_columns(pl.col("acc").fill_null(False))
pool = pool.join(rec.select(pl.col("rid").alias("q"), "src"), on="q").join(rec.select(pl.col("rid").alias("s1"), "country"), on="s1")
cnt = pool.filter(pl.col("acc")).group_by("s1", "src").agg(pl.len().alias("n_acc"))
pool = pool.join(cnt, on=["s1", "src"], how="left").with_columns(pl.col("n_acc").fill_null(0))
pool = pool.with_columns((pl.col("n_acc") - pl.col("acc").cast(pl.Int32)).alias("n_other"))
cols = {"business_name": "n", "addr_n": "a", "house": "h"}
pool = pool.join(rec.select(pl.col("rid").alias("q"), *[pl.col(k).alias(v + "q") for k, v in cols.items()]), on="q") \
           .join(rec.select(pl.col("rid").alias("s1"), *[pl.col(k).alias(v + "s") for k, v in cols.items()]), on="s1")
# classify only the records near the cap plus a sample elsewhere (classification is slow in Python)
pool = pool.with_columns(pl.col("src").replace_strict(CAP, default=6).alias("cap"))
near = pool.filter(pl.col("n_other") >= pl.col("cap") - 3)
print("records near cap:", near.height, flush=True)
near = near.with_columns(pl.Series("t", [f"{name_op(a, b)}|{addr_op(*r)}" for a, b, *r in
                                         zip(near["nq"].to_list(), near["ns"].to_list(), near["aq"].to_list(), near["hq"].to_list(), near["as"].to_list(), near["hs"].to_list())]))
near = near.with_columns((pl.col("cap") - pl.col("n_other")).clip(0, 3).alias("gap"))
# denominator: S1 x source at each gap level
s1src = pl.concat([rec.filter(pl.col("src") == 1).select(pl.col("rid").alias("s1"), "country").with_columns(pl.lit(s, dtype=pl.Int8).alias("src")) for s in (2, 3)])
den = s1src.join(cnt, on=["s1", "src"], how="left").with_columns(pl.col("n_acc").fill_null(0), pl.col("src").replace_strict(CAP, default=6).alias("cap"))
# for a record not accepted, n_other = n_acc; for accepted, n_acc - 1: use n_acc level of the S1 for denominators (approx.)
den = den.with_columns((pl.col("cap") - pl.col("n_acc")).clip(0, 3).alias("gap")).group_by("country", "gap").agg(pl.len().alias("n_s1src"))
for country in ("US", "France"):
    x = near.filter(pl.col("country") == country).group_by("t", "gap").agg(pl.len().alias("n"), pl.col("acc").mean().alias("acc_rate"))
    x = x.join(den.filter(pl.col("country") == country).drop("country"), on="gap").with_columns((pl.col("n") / pl.col("n_s1src") * 1000).alias("per1k"))
    p = x.pivot(on="gap", index="t", values="per1k").fill_null(0)
    p = p.rename({c: f"gap{c}" for c in p.columns if c != "t"})
    for c in ("gap0", "gap1", "gap2", "gap3"):
        if c not in p.columns:
            p = p.with_columns(pl.lit(0.0).alias(c))
    p = p.with_columns((pl.col("gap0") / pl.col("gap3")).round(3).alias("atcap_vs_gap3")).sort("gap3", descending=True)
    acc_rates = x.filter(pl.col("gap") == 3).select("t", pl.col("acc_rate").round(3).alias("acc_rate_gap3"))
    print(f"\n=== {country}: records per 1k (S1, source) by distance to the cap (gap0 = other records already at the cap)")
    print(p.join(acc_rates, on="t", how="left").select("t", "gap3", "gap2", "gap1", "gap0", "atcap_vs_gap3", "acc_rate_gap3").head(30))


# --- lower bound on accepted decoys per category: decoys ~ flat in n_other, so D(X) ~ gap0 rate x all (S1, source) slots;
# FP(X) >= D(X) - rejected(X). Classify all records of the country (slow): sample 25% and scale.
print("\n=== decoy lower bounds (25% sample, scaled)")
for country in ("US", "France"):
    allc = pool.filter(pl.col("country") == country).sample(fraction=0.25, seed=3)
    allc = allc.with_columns(pl.Series("t", [f"{name_op(a, b)}|{addr_op(*r)}" for a, b, *r in
                                             zip(allc["nq"].to_list(), allc["ns"].to_list(), allc["aq"].to_list(), allc["hq"].to_list(), allc["as"].to_list(), allc["hs"].to_list())]))
    tot = allc.group_by("t").agg((pl.len() * 4).alias("N"), (pl.col("acc").sum() * 4).alias("A"))
    x = near.filter(pl.col("country") == country).filter(pl.col("gap") == 0).group_by("t").agg(pl.len().alias("n0"))
    slots = den.filter(pl.col("country") == country)
    n_slots = slots["n_s1src"].sum(); n_gap0 = slots.filter(pl.col("gap") == 0)["n_s1src"].sum()
    x = x.with_columns((pl.col("n0") / n_gap0 * n_slots).alias("D"))
    y = tot.join(x, on="t", how="left").with_columns(pl.col("D").fill_null(0)).with_columns((pl.col("N") - pl.col("A")).alias("R"))
    y = y.with_columns((pl.col("D") - pl.col("R")).alias("FP_lower_bound"), (pl.col("D") / pl.col("N")).round(3).alias("decoy_share"))
    print(country, "slots", n_slots, "at-cap slots", n_gap0)
    print(y.filter(pl.col("A") > 2000).sort("FP_lower_bound", descending=True).select("t", "N", "A", "R", pl.col("D").round(0), "decoy_share", pl.col("FP_lower_bound").round(0)).head(20))

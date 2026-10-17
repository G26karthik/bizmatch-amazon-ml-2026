"""Missed-copy search with the cap. REJECTED best-S1 candidates (test_best, not accepted), classified by name op x
address op. If a class of rejected records vanishes where the S1's accepted records already fill the cap, those records
behave like copies of that S1 -> false negatives. Decoys / namesakes do not vanish. Reports per class: rejected count,
rate per 1k S1-source slots by n_other (accepted records in the source), at-cap / n0 ratio, mean model prob. US calibrates.

python scripts/cap_rejected.py /work/out_fr11
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
acc = g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls().filter(pl.col("matched_entity_ids") != "") \
       .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
       .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1")
tb = pl.read_parquet("/work/run28/test_best_run28_7q.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
rej = tb.join(acc.select("q"), on="q", how="anti").filter(pl.col("prob") >= 0.02)
rej = rej.join(rec.select(pl.col("rid").alias("q"), "src"), on="q").join(rec.select(pl.col("rid").alias("s1"), "country"), on="s1")
accs = acc.join(rec.select(pl.col("rid").alias("q"), "src"), on="q")
cnt = accs.group_by("s1", "src").agg(pl.len().alias("n_other"))
rej = rej.join(cnt, on=["s1", "src"], how="left").with_columns(pl.col("n_other").fill_null(0))
cols = {"business_name": "n", "addr_n": "a", "house": "h"}
rej = rej.join(rec.select(pl.col("rid").alias("q"), *[pl.col(k).alias(v + "q") for k, v in cols.items()]), on="q") \
         .join(rec.select(pl.col("rid").alias("s1"), *[pl.col(k).alias(v + "s") for k, v in cols.items()]), on="s1")
print("rejected candidates (p >= 0.02):", rej.group_by("country").len().rows(), flush=True)
rej = rej.with_columns(pl.Series("t", [f"{name_op(a, b)}|{addr_op(*r)}" for a, b, *r in
                                       zip(rej["nq"].to_list(), rej["ns"].to_list(), rej["aq"].to_list(), rej["hq"].to_list(), rej["as"].to_list(), rej["hs"].to_list())]))
s1s = rec.filter(pl.col("src") == 1).select(pl.col("rid").alias("s1"), "country")
slots = pl.concat([s1s.with_columns(pl.lit(s, dtype=pl.Int8).alias("src")) for s in (2, 3)]).join(cnt, on=["s1", "src"], how="left").with_columns(pl.col("n_other").fill_null(0))
den = slots.group_by("country", "src", "n_other").agg(pl.len().alias("n_slots"))
x = rej.group_by("country", "t", "src", "n_other").agg(pl.len().alias("n"), pl.col("prob").mean().alias("p")).join(den, on=["country", "src", "n_other"]) \
       .with_columns((pl.col("n") / pl.col("n_slots") * 1000).alias("per1k"))
for country in ("US", "France"):
    y = x.filter(pl.col("country") == country)
    tot = rej.filter(pl.col("country") == country).group_by("t").agg(pl.len().alias("rejected"), pl.col("prob").mean().round(3).alias("mean_p"), (pl.col("prob") * 1).sum().round(0).alias("sum_p"))
    lo = y.filter(pl.col("n_other") <= 1).group_by("t").agg(((pl.col("n")).sum() / pl.col("n_slots").sum() * 1000).alias("rate_n01"))
    cap = y.filter(((pl.col("src") == 2) & (pl.col("n_other") >= 5)) | ((pl.col("src") == 3) & (pl.col("n_other") >= 6))) \
           .group_by("t").agg(((pl.col("n")).sum() / pl.col("n_slots").sum() * 1000).alias("rate_cap"))
    mid = y.filter(pl.col("n_other").is_between(3, 4)).group_by("t").agg(((pl.col("n")).sum() / pl.col("n_slots").sum() * 1000).alias("rate_n34"))
    z = tot.join(lo, on="t", how="left").join(mid, on="t", how="left").join(cap, on="t", how="left").fill_null(0) \
           .with_columns((pl.col("rate_cap") / pl.col("rate_n34")).round(3).alias("cap_vs_n34"))
    print(f"\n=== {country}: rejected candidates by class (copy-like classes vanish at the cap)")
    print(z.sort("rejected", descending=True).head(28).with_columns([pl.col(c).round(2) for c in ("rate_n01", "rate_n34", "rate_cap")]))
rej.filter(pl.col("country") == "France").select("q", "s1", "prob", "t", "src", "n_other").write_parquet("/work/dive/fr_rejected_cls.parquet")

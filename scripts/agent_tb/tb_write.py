"""Write test picks at a fixed threshold (default 0.85) from tb_test_top.parquet, respecting 5 S2 / 6 S3 caps vs fr27.
US/India -> local_data/work/agent_tb_adds.parquet ; France (no train labels) -> agent_tb_adds_france_UNVALIDATED.parquet"""
import sys
import polars as pl
pl.Config.set_tbl_rows(-1)
thr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.85
tt = pl.read_parquet("local_data/work/tb_test_top.parquet")
g = pl.read_csv("output/v28q_fr27/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
asg = (g.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
       .filter(pl.col("matched_entity_ids") != "").select(s1="source1_entity_id", m="matched_entity_ids"))
assigned = set(asg["m"].to_list())
cnt = {}
for s1, m in asg.iter_rows():
    cnt[(s1, m[:2])] = cnt.get((s1, m[:2]), 0) + 1
un = tt.filter(pl.col("fr27_s1").is_null() & (pl.col("p") >= thr)).sort("p", descending=True)
keep, capped = [], 0
for m, s1, c, k, p in un.select("m", "s1", "country", "kind", "p").iter_rows():
    assert m not in assigned
    lim = 5 if m[:2] == "S2" else 6
    if cnt.get((s1, m[:2]), 0) >= lim:
        capped += 1; continue
    cnt[(s1, m[:2])] = cnt.get((s1, m[:2]), 0) + 1
    keep.append((m, s1, c, k, p))
out = pl.DataFrame(keep, schema={"m": pl.String, "s1": pl.String, "country": pl.String, "kind": pl.String, "p": pl.Float64}, orient="row")
assert out["m"].n_unique() == out.height
print(f"thr {thr}: picks {out.height}, dropped by cap {capped}")
print(out.group_by("country").agg(pl.len(), pl.col("p").mean().round(4).alias("mean_p(=exp.prec if calibrated)")).sort("country"))
print(out.group_by("country", "kind").len().sort("country", "kind"))
for x in (0.8, 0.85, 0.9, 0.95):
    print(" count p>=", x, out.filter(pl.col("p") >= x).group_by("country").len().sort("country").rows())
out.filter(pl.col("country") != "France").write_parquet("local_data/work/agent_tb_adds.parquet")
out.filter(pl.col("country") == "France").write_parquet("local_data/work/agent_tb_adds_france_UNVALIDATED.parquet")

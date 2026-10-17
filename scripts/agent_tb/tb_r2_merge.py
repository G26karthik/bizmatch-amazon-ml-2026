"""Merge round-2 test picks (small/big/ns120 tops): records unassigned in fr44, US/India, p >= per-country band threshold;
caps (5 S2 / 6 S3) vs fr44 applied jointly in descending p. Writes local_data/work/agent_tb_round2.parquet."""
import os
import polars as pl
pl.Config.set_tbl_rows(-1)
W = "local_data/work/"
tops = [pl.read_parquet(W + f"tb_r2_{n}_top.parquet") for n in ("small", "big", "ns120") if os.path.exists(W + f"tb_r2_{n}_top.parquet")]
tt = pl.concat(tops)
print("models:", tt["model"].unique().to_list(), "thresholds:", tt.group_by("model", "country").agg(pl.col("thr").first()).sort("model", "country").rows())
g = pl.read_csv("output/v28q_fr44/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
asg = (g.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
       .filter(pl.col("matched_entity_ids") != "").select(s1="source1_entity_id", m="matched_entity_ids"))
assigned = set(asg["m"].to_list())
cnt = {}
for s1, m in asg.iter_rows():
    cnt[(s1, m[:2])] = cnt.get((s1, m[:2]), 0) + 1
assert tt["m"].n_unique() == tt.height, "record in two group types"
un = tt.filter(~pl.col("m").is_in(list(assigned)) & pl.col("country").is_in(["US", "India"]) & (pl.col("p") >= pl.col("thr"))).sort("p", descending=True)
keep, capped = [], 0
for m, s1, c, k, p, md in un.select("m", "s1", "country", "kind", "p", "model").iter_rows():
    lim = 5 if m[:2] == "S2" else 6
    if cnt.get((s1, m[:2]), 0) >= lim:
        capped += 1; continue
    cnt[(s1, m[:2])] = cnt.get((s1, m[:2]), 0) + 1
    keep.append((m, s1, c, k, p, md))
out = pl.DataFrame(keep, schema={"m": pl.String, "s1": pl.String, "country": pl.String, "kind": pl.String, "p": pl.Float64, "model": pl.String}, orient="row")
print(f"NEW picks (unassigned in fr44): {out.height}, cap-dropped {capped}")
print(out.group_by("model", "country").agg(pl.len(), pl.col("p").mean().round(4).alias("mean_p"), pl.col("p").min().round(3).alias("min_p")).sort("model", "country"))
out.select("m", "s1", "country", "kind", "p").write_parquet(W + "agent_tb_round2.parquet")
out.write_parquet(W + "agent_tb_round2_with_model.parquet")

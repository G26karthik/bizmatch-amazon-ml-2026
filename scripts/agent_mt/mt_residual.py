"""agent_mt: residual-like eval. Train: is the OOF pick pair / owner pair in the production retrieval set
(local_data/pruned_pairs_train.parquet, rid = row index over train sources 1,2,3)? Test analog: pick pair in fr27
candidate_pairs.tsv (retrieved, scored, rejected) vs never a candidate.

uv run --no-project --with polars==1.44.2 python scripts/agent_mt/mt_residual.py
"""
import sys

import polars as pl

sys.path.insert(0, "scripts/agent_mt")
from mt_train import W, picks, report  # noqa: E402

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(250)
ids = pl.concat([pl.scan_parquet(f"local_data/train_source{s}.parquet").select("entity_id") for s in (1, 2, 3)]).with_row_index("rid").collect()
oof = pl.read_parquet(f"{W}/oof.parquet")
tp = pl.read_parquet("local_data/train_pairs.parquet")
pk = picks(oof, "p_bin").with_columns(n_named=pl.lit(0))
r = ids.rename({"entity_id": "m", "rid": "q"})
s = ids.rename({"entity_id": "s1", "rid": "srid"})
pr = pl.scan_parquet("local_data/pruned_pairs_train.parquet")
qs = r.join(oof.select("m").unique(), on="m")
sub = pr.join(qs.lazy(), on="q").collect().join(s, left_on="s1", right_on="srid").select("m", pl.col("s1_right").alias("s1c"))
subset = set(zip(sub["m"].to_list(), sub["s1c"].to_list()))
# sanity: truth pairs coverage by retrieval, over OOF records
own = dict(zip(tp["m"].to_list(), tp["s1"].to_list()))
cov = [(m_, own[m_]) in subset for m_ in oof["m"].unique().to_list() if m_ in own]
print("retrieval coverage of owner pair for OOF records:", round(sum(cov) / len(cov), 4), len(cov))
pk = pk.with_columns(pick_in=pl.Series([(a, b) in subset for a, b in zip(pk["m"].to_list(), pk["s1"].to_list())]),
                     own_in=pl.Series([(a, own.get(a)) in subset for a in pk["m"].to_list()]))
print(pk.group_by("pick_in", "own_in").agg(pl.len(), pl.col("label").mean().round(4)).sort("pick_in", "own_in"))
report(pk.filter(~pl.col("own_in")), "p_bin", "OOF, owner pair NOT retrieved (main model cannot have it)")
report(pk.filter(~pl.col("pick_in")), "p_bin", "OOF, pick pair NOT retrieved (analog of test 'never a candidate')")
report(pk.filter(~pl.col("pick_in") & ~pl.col("own_in")), "p_bin", "OOF, neither pick nor owner retrieved")
pk.select("m", "s1", "pick_in", "own_in").write_parquet(f"{W}/oof_retr.parquet")

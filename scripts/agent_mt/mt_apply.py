"""agent_mt: test-like subset eval of OOF picks + apply saved model to test unassigned multi-tenant one-token rows.

uv run --no-project --with polars==1.44.2 --with lightgbm --with numpy python scripts/agent_mt/mt_apply.py [thr]
"""
import sys

import lightgbm as lgb
import polars as pl

sys.path.insert(0, "scripts/agent_mt")
from mt_train import W, feats, picks, prep, report  # noqa: E402

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(250)
THR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.85


def flags(d):
    return d.with_columns(n_named=(pl.col("kind") != "pseudo").sum().over("gid"))


oof = flags(pl.read_parquet(f"{W}/oof.parquet"))
pk = picks(oof, "p_bin")
report(pk.filter(pl.col("n_named") != 1), "p_bin", "OOF, groups NOT covered by unique-named rule (n_named != 1)")
report(pk.filter(pl.col("n_named") == 0), "p_bin", "OOF, all-pseudo groups (n_named == 0)")
report(pk.filter(pl.col("n_named") >= 2), "p_bin", "OOF, ambiguous-named groups (n_named >= 2)")
report(pk.filter(pl.col("pr") < 80), "p_bin", "OOF, pick partial_ratio < 80")
report(pk.filter((pl.col("pr") < 70) & (pl.col("kind") == "pseudo")), "p_bin", "OOF, pseudo pick with partial_ratio < 70")

mb = lgb.Booster(model_file=f"{W}/mt_bin.txt")
F = feats()
te = flags(prep(pl.read_parquet(f"{W}/rows_test.parquet")))
te = te.with_columns(p_bin=pl.Series(mb.predict(te.select(F).to_numpy())))
te.write_parquet(f"{W}/test_scored.parquet")
tpk = picks(te, "p_bin")
cp = (pl.scan_csv("output/v28q_fr27/candidate_pairs.tsv", separator="	", schema_overrides={"source1_entity_id": pl.String, "candidate_entity_ids": pl.String})
      .filter(pl.col("source1_entity_id").is_in(tpk["s1"].unique().implode())).collect()
      .select(s1=pl.col("source1_entity_id"), m=pl.col("candidate_entity_ids").fill_null("").str.split(",")).explode("m")
      .with_columns(pick_in=pl.lit(True)))
tpk = tpk.join(cp, on=["s1", "m"], how="left").with_columns(pl.col("pick_in").fill_null(False))
print("\nTEST picks: pick pair retrieved by fr27 (scored & rejected) vs never a candidate:")
print(tpk.group_by("country", "pick_in").agg(pl.len(), (pl.col("p_bin") >= 0.85).sum().alias("p85"), pl.col("p_bin").mean().round(3)).sort("country", "pick_in"))
print("\nTEST groups / picks by country, n_named:")
print(tpk.group_by("country", "n_named").agg(pl.len(), (pl.col("p_bin") >= 0.8).sum().alias("p80"), (pl.col("p_bin") >= 0.85).sum().alias("p85"),
                                             (pl.col("p_bin") >= 0.9).sum().alias("p90"), pl.col("p_bin").mean().round(3).alias("mean_p")).sort("country", "n_named"))
print(tpk.filter(pl.col("p_bin") >= THR).group_by("country", "kind").agg(pl.len(), pl.col("pr").mean().round(1)).sort("country", "kind"))
# feature-shift check vs train picks
print("\npick feature means train-OOF vs test (US/India):")
cols = ["pr", "pr_dmean", "pre_dmean", "jf_dmean", "sub_dmean", "cp_tot", "cp_same", "tok_len", "n_cand", "p_bin"]
print(pl.concat([pk.select(pl.lit("train").alias("set"), "country", *cols), tpk.filter(pl.col("country") != "France").select(pl.lit("test").alias("set"), "country", *cols)])
      .group_by("set", "country").agg([pl.col(c).mean().round(3) for c in cols] + [pl.len()]).sort("country", "set"))

# caps vs fr27
g = pl.read_csv("output/v28q_fr27/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
ex = g.select(s1=pl.col("source1_entity_id"), m=pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("m").filter(pl.col("m") != "")
cnt = ex.with_columns(src=pl.col("m").str.slice(0, 2)).group_by("s1", "src").len()
sel = tpk.filter((pl.col("p_bin") >= THR) & (~pl.col("pick_in") | ("--allin" in sys.argv))).select("m", "s1", "country", "kind", pl.col("p_bin").alias("p"), "src")
sel = sel.join(cnt, on=["s1", "src"], how="left").with_columns(pl.col("len").fill_null(0))
# add in descending p within (s1, src), respect caps 5/6
sel = sel.sort("p", descending=True).with_columns(k=pl.int_range(pl.len()).over("s1", "src") + 1)
sel = sel.with_columns(ok=(pl.col("len") + pl.col("k")) <= pl.when(pl.col("src") == "S2").then(5).otherwise(6))
print("\ncap check: dropped", sel.filter(~pl.col("ok")).height, "of", sel.height)
sel = sel.filter(pl.col("ok"))
usin = sel.filter(pl.col("country") != "France").select("m", "s1", "country", "kind", "p")
fr = sel.filter(pl.col("country") == "France").select("m", "s1", "country", "kind", "p")
print("adds US/India:", usin.group_by("country", "kind").len().sort("country", "kind"))
print("France (unvalidated, NOT in main file):", fr.group_by("kind").len().sort("kind"))
if "--write" in sys.argv:
    usin.write_parquet("local_data/work/agent_mt_adds.parquet")
    fr.write_parquet("local_data/work/agent_mt_adds_france_unvalidated.parquet")
    print("wrote", usin.height, fr.height)

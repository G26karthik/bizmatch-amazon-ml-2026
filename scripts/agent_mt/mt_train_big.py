"""agent_mt BIG: LightGBM tie-breaker for mode big (4-8 same-house S1s) and mode street (same rare-street-token set, different
or missing house number, 1-8 S1s). 2-fold OOF split by crc32(record address key); per-mode, per-p-band precision with the
address/name tie guard; production-residual check (val split, local_data/llm/unc_val.parquet); apply to test (fr30 unassigned).

uv run --no-project --with polars==1.44.2 --with lightgbm --with numpy python scripts/agent_mt/mt_train_big.py [THR] [--write]
"""
import sys
import zlib

import lightgbm as lgb
import polars as pl

pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(40)
W = "local_data/work/agent_mt"
THR = float(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else 0.9
BASE = ["n_cand", "tok_len", "tok_alpha", "tok_digit", "tok_dot", "js", "jf", "cp_same", "cp_oth", "cp_tot", "cp_same_room",
        "al_same", "al_oth", "compet", "addr_exact", "n_q_addr", "initials", "ini_prefix", "pr", "prw", "rat", "handle", "sub",
        "sub_frac", "pre", "first", "legal", "c_ntok", "c_nws", "c_len", "is_s3", "ctry_us",
        "mode_street", "hn_eq", "rec_has_h", "cand_has_h", "hn_share"]
REL = ["pr", "prw", "rat", "sub", "pre", "initials", "handle", "cp_tot", "cp_same", "js", "jf", "c_len", "hn_share"]
F = BASE + [f"{c}_{s}" for c in REL for s in ("dmax", "dmean", "rk")]
PAR = dict(learning_rate=0.03, num_leaves=15, min_data_in_leaf=30, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
           lambda_l2=1.0, verbose=-1, num_threads=4, seed=7)


def prep(d):
    d = d.with_columns(is_s3=(pl.col("src") == "S3").cast(pl.Int8), ctry_us=(pl.col("country") == "US").cast(pl.Int8), gid=pl.col("m"))
    ex = []
    for c in REL:
        ex += [(pl.col(c) - pl.col(c).max().over("gid")).alias(f"{c}_dmax"),
               pl.when(pl.col("n_cand") > 1).then(pl.col(c) - (pl.col(c).sum().over("gid") - pl.col(c)) / (pl.col("n_cand") - 1)).otherwise(0.0).alias(f"{c}_dmean"),
               pl.col(c).rank("min", descending=True).over("gid").alias(f"{c}_rk")]
    d = d.with_columns(ex).sort("gid", "s1")
    d = d.with_columns(jfmax=pl.col("jf").max().over("gid"), prmax=pl.col("pr").max().over("gid"))
    return d.with_columns(n_jfmax=(pl.col("jf") == pl.col("jfmax")).sum().over("gid"),
                          n_prtop=((pl.col("pr") >= 90) & (pl.col("pr") == pl.col("prmax"))).sum().over("gid"))


def picks(d, p):
    pk = d.sort(p, descending=True).group_by("gid", maintain_order=True).first()
    return pk.with_columns(tie_addr=(pl.col("jf") == pl.col("jfmax")) & (pl.col("n_jfmax") >= 2), tie_name=(pl.col("n_prtop") >= 2) & (pl.col("pr") >= 90))


def fit(tr):
    return lgb.train({**PAR, "objective": "binary"}, lgb.Dataset(tr.select(F).to_numpy(), tr["label"].to_numpy()), num_boost_round=400)


def bands(pk, title):
    print(f"\n== {title}: groups {pk.height}, owner-in-cands {pk['has_pos'].mean():.4f}, top1 {pk['label'].mean():.4f}")
    b = pk.filter(pl.col("p_bin") >= 0.7).with_columns(bin=pl.col("p_bin").cut([0.8, 0.85, 0.9, 0.95, 0.98], left_closed=True))
    print(b.group_by("bin").agg(pl.len(), prec=pl.col("label").mean().round(4), noise=(pl.col("m_has_owner") == 0).sum()).sort("bin"))
    for t in (0.85, 0.9, 0.95):
        z = pk.filter(pl.col("p_bin") >= t)
        print(f"  cum p>={t}: n={z.height} cov={z.height / max(1, pk.height):.3f} prec={z['label'].mean() if z.height else float('nan'):.4f}")


tr = prep(pl.read_parquet(f"{W}/rows_big_train.parquet"))
tr = tr.with_columns(fold=pl.col("akey").map_elements(lambda s: zlib.crc32(s.encode()) % 2, return_dtype=pl.Int64), has_pos=pl.col("label").max().over("gid"))
print(tr.group_by("country", "mode_street").agg(groups=pl.col("gid").n_unique(), rows=pl.len(), owner=pl.col("m_has_owner").mean().round(4),
                                                 has_pos=pl.col("has_pos").mean().round(4)).sort("country", "mode_street"))
parts = []
for k in (0, 1):
    a, b = tr.filter(pl.col("fold") != k), tr.filter(pl.col("fold") == k)
    parts.append(b.with_columns(p_bin=pl.Series(fit(a).predict(b.select(F).to_numpy()))))
oof = pl.concat(parts)
pk = picks(oof, "p_bin")
for ms in (0, 1):
    for gd in (False, True):
        s = pk.filter(pl.col("mode_street") == ms)
        if gd:
            s = s.filter(~pl.col("tie_addr") & ~pl.col("tie_name"))
        bands(s, f"OOF mode={'street' if ms else 'big'} {'GUARDED' if gd else 'unguarded'}")
    s = pk.filter((pl.col("mode_street") == ms) & ~pl.col("tie_addr") & ~pl.col("tie_name") & (pl.col("pr_dmean") < 10) & (pl.col("pre_dmean") < 2) & (pl.col("sub_dmean") < 3))
    bands(s, f"OOF mode={'street' if ms else 'big'} GUARDED residual-like (no name contrast)")
pk.write_parquet(f"{W}/oof_big_picks.parquet")

# production residual check on the testlike_rule val split
SEED, VAL_MOD, HIDE = 42, 10, 19
rec = pl.concat([pl.scan_parquet(f"local_data/train_source{s}.parquet").select("entity_id").with_columns(pl.lit(s).cast(pl.Int8).alias("src_n")) for s in (1, 2, 3)]).with_row_index("rid").collect()
rec = rec.with_columns(hidden=(pl.col("src_n") == 1) & ((pl.col("entity_id").hash(SEED + 77) % 100) < HIDE), own_val=pl.col("entity_id").hash(SEED) % VAL_MOD == 0)
tp = pl.read_parquet("local_data/train_pairs.parquet").join(pk.select("m"), on="m")
s1i = rec.filter(pl.col("src_n") == 1).select(pl.col("entity_id").alias("owner"), pl.col("hidden").alias("own_hidden"), pl.col("own_val").alias("par_val"))
v = pk.join(tp.rename({"s1": "owner"}), on="m", how="left").join(s1i, on="owner", how="left")
v = v.join(rec.select(pl.col("entity_id").alias("m"), pl.col("rid").alias("q"), pl.col("own_val").alias("m_val")), on="m", how="left")
v = v.with_columns(is_val=pl.coalesce("par_val", "m_val")).filter(pl.col("is_val") & ~pl.col("own_hidden").fill_null(False))
uv = pl.read_parquet("local_data/llm/unc_val.parquet").select(pl.col("q").cast(pl.UInt32), pl.col("prob").cast(pl.Float64)).group_by("q").agg(pmax=pl.col("prob").max())
v = v.join(uv, on="q", how="left")
print("\nval groups:", v.height)
for ms in (0, 1):
    for thr_acc in (0.8, 0.9):
        res = v.filter((pl.col("mode_street") == ms) & ((pl.col("pmax") < thr_acc) | (pl.col("pmax").is_null() & pl.col("owner").is_null())))
        g = res.filter(~pl.col("tie_addr") & ~pl.col("tie_name"))
        out = [f"prod-residual mode={'street' if ms else 'big'} pmax<{thr_acc}: groups {res.height} (noise {(res['m_has_owner'] == 0).sum()})"]
        for t in (0.85, 0.9, 0.95):
            z, zg = res.filter(pl.col("p_bin") >= t), g.filter(pl.col("p_bin") >= t)
            out.append(f"p>={t}: {z['label'].sum()}/{z.height} guarded {zg['label'].sum()}/{zg.height}")
        print(" | ".join(out))

# test
mb = fit(tr)
te = prep(pl.read_parquet(f"{W}/rows_big_test.parquet"))
te = te.with_columns(p_bin=pl.Series(mb.predict(te.select(F).to_numpy())))
tpk = picks(te, "p_bin")
tpk.write_parquet(f"{W}/test_big_picks.parquet")
print("\nTEST groups by country/mode, picks at thresholds (guarded):")
tg = tpk.filter(~pl.col("tie_addr") & ~pl.col("tie_name"))
print(tpk.group_by("country", "mode_street").agg(pl.len(), pl.col("p_bin").mean().round(3)).sort("country", "mode_street"))
print(tg.group_by("country", "mode_street").agg(pl.len(), (pl.col("p_bin") >= 0.85).sum().alias("p85"), (pl.col("p_bin") >= 0.9).sum().alias("p90"),
                                               (pl.col("p_bin") >= 0.95).sum().alias("p95")).sort("country", "mode_street"))
MODES = [0, 1] if "--street" in sys.argv else [0] if "--bigonly" in sys.argv else [0, 1]
sel = tg.filter((pl.col("p_bin") >= THR) & pl.col("mode_street").is_in(MODES)).with_columns(
    kind=pl.format("{}_{}", pl.when(pl.col("mode_street") == 1).then(pl.lit("street")).otherwise(pl.lit("big")), pl.col("kind"))).select("m", "s1", "country", "kind", pl.col("p_bin").alias("p"), "src")
g = pl.read_csv("output/v28q_fr30/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
ex = g.select(s1=pl.col("source1_entity_id"), m=pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("m").filter(pl.col("m") != "")
assert sel.join(ex, on="m").height == 0
cnt = ex.with_columns(src=pl.col("m").str.slice(0, 2)).group_by("s1", "src").len()
sel = sel.join(cnt, on=["s1", "src"], how="left").with_columns(pl.col("len").fill_null(0)).sort("p", descending=True)
sel = sel.with_columns(k=pl.int_range(pl.len()).over("s1", "src") + 1)
sel = sel.with_columns(ok=(pl.col("len") + pl.col("k")) <= pl.when(pl.col("src") == "S2").then(5).otherwise(6))
print("cap drops:", sel.filter(~pl.col("ok")).height)
sel = sel.filter(pl.col("ok")).select("m", "s1", "country", "kind", "p")
print(sel.group_by("country", "kind").agg(pl.len(), pl.col("p").min().round(3)).sort("country", "kind"))
print(te.join(sel.select("m", "s1"), on=["m", "s1"]).select("country", "mname", "sname", "p_bin", "mode_street", "n_cand", "jf_dmean", "hn_eq").sort("p_bin").head(20))
if "--write" in sys.argv:
    sel.write_parquet("local_data/work/agent_mt_adds_big.parquet")
    print("wrote", sel.height)

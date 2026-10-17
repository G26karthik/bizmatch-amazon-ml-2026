"""E00: where is the remaining validation loss (run28_7q, half B) and how much is recoverable.

Stage oracles (mean over half-B S1s of 5m/(T+4m), m = true copies still present), Shapley decomposition of the actual
per-S1 loss into error categories, slices, and a name+address recoverability test for every missed / false record.

Phase 1 (no retrieval file): 'not_cand' = true pair absent from the pruned set (pA >= 0.001); writes
/work/agent_e00/need_retr.parquet for scripts/e00_retr.py. Phase 2 (with /work/agent_e00/retr_ranks.parquet): splits
'not_cand' into 'not_retrieved' and 'pruned_A<0.001'.

cd /work/code && PYTHONPATH=src python /work/agent_e00/e00_loss.py
"""
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "src")
from ber.metric import entity_f05  # noqa: E402
from ber.pipeline import SEED, hide_pct, hidden_s1, labels  # noqa: E402
from ber.stage2 import _chosen  # noqa: E402

OUT = Path("/work/agent_e00")
OUT.mkdir(exist_ok=True)
work, tag = Path("/work/run28"), "run28_7q"
cfg = {"mode": "efopt", "miss": 0.0, "gamma": 1.5}
pct = hide_pct(work)

# ------------------------------------------------------------------ labels exactly as scripts/eval_judge.py
lab = labels(Path("/work/data"), work).with_columns((pl.col("parent").fill_null(pl.col("rid")).hash(SEED + 7) % 2 == 0).alias("half_a"))
SB = lab.filter((pl.col("src") == 1) & pl.col("is_val") & ~pl.col("half_a"))["rid"].to_list()
SBset = set(SB)
NB = len(SB)
par_df = lab.filter(pl.col("parent").is_not_null()).select(pl.col("rid").alias("q"), pl.col("parent").alias("s1"))
parent = dict(par_df.iter_rows())
truth = {}
for q, p in parent.items():
    truth.setdefault(p, set()).add(q)
orph = set(lab.filter(pl.col("parent").is_null() & (pl.col("src") != 1))["rid"].to_list())
gt = pl.read_parquet("/work/data/train_pairs.parquet")
Hm = gt.filter(hidden_s1(pl.col("s1"), pct)).select(pl.col("m").alias("entity_id"))
H = set(lab.join(Hm, on="entity_id", how="semi")["rid"].to_list())
print(f"half B S1s {NB}; orphans {len(orph)} of which hidden-parent copies {len(H & orph)}", flush=True)

vb = pl.read_parquet(work / f"val_best_{tag}.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
ch = _chosen(vb, cfg)
pred = {}
for q, s in ch.select("q", "s1").iter_rows():
    pred.setdefault(s, set()).add(q)
chosen_of = dict(ch.select("q", "s1").iter_rows())  # q -> s1 it was assigned to
am = {q: (s, p) for q, s, p in vb.select("q", "s1", "prob").iter_rows()}
Fb = np.array([entity_f05(truth.get(s, ()), pred.get(s, ())) for s in SB])
print(f"half B plain F0.5 {Fb.mean():.6f} (target 0.991316); loss {1 - Fb.mean():.6f} = {(1 - Fb).sum():.1f} S1-units", flush=True)

# ------------------------------------------------------------------ stage sets for half-B true pairs
tp = pl.DataFrame({"s1": [s for s in SB for _ in truth.get(s, ())], "q": [q for s in SB for q in truth.get(s, ())]},
                  schema={"s1": pl.UInt32, "q": pl.UInt32})
prn = pl.concat([pl.read_parquet(p, columns=["q", "s1", "pA", "n_cand"]) for p in sorted((work / "train" / "pruned").glob("*.parquet"))])
prn = prn.with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
sv = pl.read_parquet(work / "scores_val.parquet", columns=["q", "s1", "p3"]).with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
zone = pl.read_parquet("/work/ce_wide/ce_scores_val.parquet", columns=["q", "s1"]).with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
t = (tp.join(prn.select("q", "s1", "pA"), on=["q", "s1"], how="left")
       .join(sv.select("q", "s1", pl.lit(True).alias("in_sv")), on=["q", "s1"], how="left")
       .join(zone.with_columns(pl.lit(True).alias("in_zone")), on=["q", "s1"], how="left")
       .with_columns(pl.col("in_sv").fill_null(False), pl.col("in_zone").fill_null(False), pl.col("pA").is_not_null().alias("in_pr")))
t = t.with_columns(pl.Series("am_s1", [am.get(q, (None, None))[0] for q in t["q"].to_list()], dtype=pl.UInt32),
                   pl.Series("am_p", [am.get(q, (None, None))[1] for q in t["q"].to_list()], dtype=pl.Float32),
                   pl.Series("ch_s1", [chosen_of.get(q) for q in t["q"].to_list()], dtype=pl.UInt32))
t = t.with_columns((pl.col("am_s1") == pl.col("s1")).fill_null(False).alias("in_am"),
                   (pl.col("ch_s1") == pl.col("s1")).fill_null(False).alias("chosen"))
print("consistency: in_sv & ~pA>=0.01:", t.filter(pl.col("in_sv") & ~(pl.col("pA") >= 0.01)).height,
      "| pA>=0.01 & ~in_sv:", t.filter((pl.col("pA") >= 0.01) & ~pl.col("in_sv")).height,
      "| zone & ~sv:", t.filter(pl.col("in_zone") & ~pl.col("in_sv")).height, flush=True)

retr_path = OUT / "retr_ranks.parquet"
if retr_path.exists():
    rr = pl.read_parquet(retr_path).with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
    t = t.join(rr.select("q", "s1", "retrieved", "min_rank", "rank_c", "rank_a", "rank_n", "in_e5"), on=["q", "s1"], how="left")
    t = t.with_columns(pl.when(pl.col("in_pr")).then(True).otherwise(pl.col("retrieved")).alias("retrieved"))
    print("retrieval info missing for not-pruned pairs:", t.filter(~pl.col("in_pr") & pl.col("retrieved").is_null()).height, flush=True)
    t = t.with_columns(pl.col("retrieved").fill_null(False))
else:
    t = t.with_columns(pl.lit(None, dtype=pl.Boolean).alias("retrieved"))
    t.filter(~pl.col("in_pr")).select("q", "s1").write_parquet(OUT / "need_retr.parquet")
    # controls: true pairs that are in the pruned set (retrieval reconstruction check)
    ctl = t.filter(pl.col("in_pr")).sample(3000, seed=1).select("q", "s1")
    ctl.write_parquet(OUT / "ctl_retr.parquet")
    print(f"wrote need_retr ({t.filter(~pl.col('in_pr')).height}) and ctl_retr (3000)", flush=True)
have_retr = retr_path.exists()

T_of = {s: len(truth.get(s, ())) for s in SB}


def oracle(present_count: dict) -> float:
    v = []
    for s in SB:
        T = T_of[s]
        m = present_count.get(s, 0)
        v.append(1.0 if T == 0 else 5 * m / (T + 4 * m))
    return float(np.mean(v))


def cnt(mask) -> dict:
    return dict(t.filter(mask).group_by("s1").len().iter_rows())


n_true = t.height
stages = []
if have_retr:
    stages.append(("(a) after retrieval (TF-IDF x3 + e5)", pl.col("retrieved")))
stages += [("(b0) after Stage-A, pA >= 0.001 (pruned files)", pl.col("in_pr")),
           ("(b) after Stage-A, pA >= 0.01 (scored set)", pl.col("in_sv")),
           ("(c) in cross-encoder zone (ce_wide scored)", pl.col("in_zone")),
           ("(d) record's argmax S1 = true S1 (val_best)", pl.col("in_am")),
           ("(e) actually chosen (decision output)", pl.col("chosen"))]
rep = {"NB": NB, "F_actual": float(Fb.mean()), "n_true_pairs": n_true, "stages": []}
print("\n=== Stage oracle (half B) ===")
print(f"{'stage':62s} {'recall':>8s} {'oracle F':>9s} {'loss vs 1':>9s}")
for name, m in stages:
    c = cnt(m)
    o = oracle(c)
    rec_ = t.filter(m).height / n_true
    rep["stages"].append({"stage": name, "recall": rec_, "oracle": o})
    print(f"{name:62s} {rec_:8.5f} {o:9.6f} {1 - o:9.6f}")
print(f"{'actual decision output (with FPs)':62s} {'':8s} {Fb.mean():9.6f} {1 - Fb.mean():9.6f}")

# ------------------------------------------------------------------ categories
if have_retr:
    notcand = pl.when(~pl.col("retrieved")).then(pl.lit("FN_not_retrieved")).otherwise(pl.lit("FN_pruned_A<0.001"))
else:
    notcand = pl.lit("FN_not_cand")
t = t.with_columns(
    pl.when(pl.col("chosen")).then(pl.lit("TP"))
      .when(~pl.col("in_pr")).then(notcand)
      .when(~pl.col("in_sv")).then(pl.lit("FN_pruned_A_0.001-0.01"))
      .when(~pl.col("in_am") & pl.col("ch_s1").is_not_null()).then(pl.lit("FN_wrong_S1_assigned"))
      .when(~pl.col("in_am")).then(pl.lit("FN_argmax_other_S1_unassigned"))
      .otherwise(pl.lit("FN_not_chosen")).alias("cat"))
fp = ch.filter(pl.col("s1").is_in(pl.Series(SB, dtype=pl.UInt32).implode())).select("q", "s1")
fp = fp.with_columns(pl.Series("tpar", [parent.get(q) for q in fp["q"].to_list()], dtype=pl.UInt32))
fp = fp.filter(pl.col("tpar").is_null() | (pl.col("tpar") != pl.col("s1")))
fp = fp.with_columns(pl.Series("isH", [q in H for q in fp["q"].to_list()]))
fp = fp.with_columns(pl.when(pl.col("tpar").is_not_null()).then(pl.lit("FP_other_S1_copy"))
                       .when(pl.col("isH")).then(pl.lit("FP_orphan_hidden_parent"))
                       .otherwise(pl.lit("FP_orphan_unmatched")).alias("cat"))
fp = fp.join(vb.select("q", pl.col("prob").alias("am_p")), on="q", how="left")
FN_CATS = sorted(set(t["cat"].unique().to_list()) - {"TP"})
FP_CATS = sorted(set(fp["cat"].unique().to_list()))
CATS = FN_CATS + FP_CATS

# per-S1 counts
tp_cnt = cnt(pl.col("cat") == "TP")
fn_cnt = {}
for s, c, n in t.filter(pl.col("cat") != "TP").group_by("s1", "cat").len().iter_rows():
    fn_cnt.setdefault(s, {})[c] = n
fp_cnt = {}
for s, c, n in fp.group_by("s1", "cat").len().iter_rows():
    fp_cnt.setdefault(s, {})[c] = n


def f05(tp_, fp_, fn_, T):
    if T == 0:
        return 1.0 if fp_ == 0 else 0.0
    return 1.25 * tp_ / (1.25 * tp_ + fp_ + 0.25 * fn_) if tp_ else 0.0


shap_tot = {c: 0.0 for c in CATS}
fixonly_tot = {c: 0.0 for c in CATS}
shap_rec = {}  # (s1, cat) -> Shapley value of that category on that S1
chk = 0.0
for s in SB:
    fnc, fpc = fn_cnt.get(s, {}), fp_cnt.get(s, {})
    if not fnc and not fpc:
        continue
    T = T_of[s]
    players = list(fnc) + list(fpc)
    n = len(players)

    def v(S):
        tp_ = tp_cnt.get(s, 0) + sum(fnc[c] for c in S if c in fnc)
        fn_ = sum(fnc[c] for c in fnc if c not in S)
        fp_ = sum(fpc[c] for c in fpc if c not in S)
        return f05(tp_, fp_, fn_, T)

    base = v(())
    vals = {}
    for r in range(n + 1):
        for S in itertools.combinations(players, r):
            vals[frozenset(S)] = v(S)
    for c in players:
        others = [x for x in players if x != c]
        sv_ = 0.0
        for r in range(n):
            w = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
            for S in itertools.combinations(others, r):
                fs = frozenset(S)
                sv_ += w * (vals[fs | {c}] - vals[fs])
        shap_tot[c] += sv_
        shap_rec[(s, c)] = sv_
        fixonly_tot[c] += vals[frozenset([c])] - base
    chk += vals[frozenset(players)] - base
print(f"\nShapley check: sum {chk / NB:.6f} vs loss {1 - Fb.mean():.6f}")
n_rec = dict(t.filter(pl.col("cat") != "TP").group_by("cat").len().iter_rows()) | dict(fp.group_by("cat").len().iter_rows())
n_s1 = {c: sum(1 for (s, cc) in shap_rec if cc == c) for c in CATS}
print("\n=== Loss breakdown (half B, F0.5 units = sum over S1 / NB) ===")
print(f"{'category':34s} {'records':>8s} {'S1s':>6s} {'Shapley':>9s} {'share':>6s} {'fix-only':>9s}")
L = 1 - Fb.mean()
rep["breakdown"] = {}
for c in sorted(CATS, key=lambda c: -shap_tot[c]):
    print(f"{c:34s} {n_rec.get(c, 0):8d} {n_s1[c]:6d} {shap_tot[c] / NB:9.6f} {shap_tot[c] / NB / L:6.1%} {fixonly_tot[c] / NB:9.6f}")
    rep["breakdown"][c] = {"records": n_rec.get(c, 0), "s1s": n_s1[c], "shapley": shap_tot[c] / NB, "fix_only": fixonly_tot[c] / NB}

# ------------------------------------------------------------------ record-level attributes
rec = pl.read_parquet(work / "train" / "records.parquet",
                      columns=["entity_id", "country", "src", "name_n", "addr_n", "addr_empty", "business_name", "business_address"]
                      ).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
pool = rec.filter((pl.col("src") == 1) & ~hidden_s1(pl.col("entity_id"), pct))
cn = pool.group_by("country", "name_n").len().rename({"len": "name_cnt"})
ca = pool.filter(pl.col("addr_n") != "").group_by("country", "addr_n").len().rename({"len": "addr_cnt"})
cna = pool.filter(pl.col("addr_n") != "").group_by("country", "name_n", "addr_n").len().rename({"len": "na_cnt"})

err = pl.concat([
    t.filter(pl.col("cat") != "TP").select("q", "s1", "cat", "am_s1", "am_p", "pA"),
    fp.select("q", "s1", "cat", pl.col("s1").alias("am_s1"), "am_p", pl.lit(None, dtype=pl.Float32).alias("pA"))])
err = err.with_columns(pl.Series("shap", [shap_rec.get((s, c), 0.0) for s, c in zip(err["s1"].to_list(), err["cat"].to_list())]))
k = err.group_by("s1", "cat").len().rename({"len": "_k"})
err = err.join(k, on=["s1", "cat"]).with_columns((pl.col("shap") / pl.col("_k")).alias("loss")).drop("_k")  # equal share per record
qa = rec.select(pl.col("rid").alias("q"), pl.col("src").alias("q_src"), pl.col("name_n").alias("q_name"),
                pl.col("addr_n").alias("q_addr"), pl.col("addr_empty").alias("q_addr_empty"), pl.col("country"),
                pl.col("business_name").alias("q_raw_name"), pl.col("business_address").alias("q_raw_addr"))
sa = rec.select(pl.col("rid").alias("s1"), pl.col("name_n").alias("s_name"), pl.col("addr_n").alias("s_addr"),
                pl.col("business_name").alias("s_raw_name"), pl.col("business_address").alias("s_raw_addr"))
err = err.join(qa, on="q", how="left").join(sa, on="s1", how="left")
err = err.with_columns(pl.Series("T", [T_of[s] for s in err["s1"].to_list()]))
err = (err.join(cn.rename({"name_n": "s_name", "name_cnt": "s_name_cnt"}), on=["country", "s_name"], how="left")
          .join(cn.rename({"name_n": "q_name", "name_cnt": "q_name_cnt"}), on=["country", "q_name"], how="left")
          .join(ca.rename({"addr_n": "q_addr", "addr_cnt": "q_addr_cnt"}), on=["country", "q_addr"], how="left")
          .join(cna.rename({"name_n": "q_name", "addr_n": "q_addr", "na_cnt": "q_na_cnt"}), on=["country", "q_name", "q_addr"], how="left")
          .with_columns(pl.col("s_name_cnt", "q_name_cnt", "q_addr_cnt", "q_na_cnt").fill_null(0)))
# true parent of the error record (for FN = s1; for FP = its real parent or None)
err = err.with_columns(pl.Series("true_s1", [s if c.startswith("FN") else parent.get(q)
                                             for q, s, c in zip(err["q"].to_list(), err["s1"].to_list(), err["cat"].to_list())], dtype=pl.UInt32))

# ------------------------------------------------------------------ name+address competitor test for FN records
from rapidfuzz import fuzz  # noqa: E402
from rapidfuzz.process import cpdist  # noqa: E402

fnr = err.filter(pl.col("cat").str.starts_with("FN")).select("q", "s1", "country", "q_name", "q_addr", "q_addr_empty")
# analogues: true pairs that WERE chosen (TP), all empty-address ones + a 25k sample of address ones (weight 1/rate)
tpall = t.filter(pl.col("cat") == "TP").select("q", "s1").join(qa.select("q", "country", "q_name", "q_addr", "q_addr_empty"), on="q")
tp_e = tpall.filter(pl.col("q_addr_empty")).with_columns(pl.lit(1.0).alias("w"))
tp_a_all = tpall.filter(~pl.col("q_addr_empty"))
tp_a = tp_a_all.sample(min(25000, tp_a_all.height), seed=2).with_columns(pl.lit(tp_a_all.height / min(25000, tp_a_all.height)).alias("w"))
tgt = pl.concat([fnr.with_columns(pl.lit(1.0).alias("w")), tp_e, tp_a])
qset = pl.Series(tgt["q"].unique().to_list(), dtype=pl.UInt32)
comp = [prn.filter(pl.col("q").is_in(qset.implode())).select("q", pl.col("s1").alias("c")),
        tgt.select("q", "country", "q_name").join(pool.select(pl.col("rid").alias("c"), "country", pl.col("name_n").alias("q_name")),
                                                  on=["country", "q_name"]).select("q", "c").group_by("q").head(60),
        tgt.filter(pl.col("q_addr") != "").select("q", "country", "q_addr")
           .join(pool.select(pl.col("rid").alias("c"), "country", pl.col("addr_n").alias("q_addr")), on=["country", "q_addr"])
           .select("q", "c").group_by("q").head(60),
        tgt.select("q", pl.col("s1").alias("c"))]
if (OUT / "retr_neighbours.parquet").exists():
    comp.append(pl.read_parquet(OUT / "retr_neighbours.parquet").select(pl.col("q").cast(pl.UInt32), pl.col("c").cast(pl.UInt32))
                .filter(pl.col("q").is_in(qset.implode())))
comp = pl.concat([c.select(pl.col("q").cast(pl.UInt32), pl.col("c").cast(pl.UInt32)) for c in comp]).unique()
print(f"\ncompetitor pairs for {qset.len()} records (FN + TP analogues): {comp.height}", flush=True)
comp = comp.join(rec.select(pl.col("rid").alias("q"), pl.col("name_n").alias("qn"), pl.col("addr_n").alias("qa")), on="q") \
           .join(rec.select(pl.col("rid").alias("c"), pl.col("name_n").alias("cn"), pl.col("addr_n").alias("ca")), on="c")
qn, cn_, qad, cad = (comp[c].to_list() for c in ("qn", "cn", "qa", "ca"))
# name: edit similarity + token-sort (token-set would tie every sub/superset name at 100)
nsim = 0.5 * cpdist(qn, cn_, scorer=fuzz.ratio, workers=2) + 0.5 * cpdist(qn, cn_, scorer=fuzz.token_sort_ratio, workers=2)
asim = 0.5 * cpdist(qad, cad, scorer=fuzz.token_set_ratio, workers=2) + 0.5 * cpdist(qad, cad, scorer=fuzz.ratio, workers=2)
qempty = (comp["qa"] == "").to_numpy()
sc = np.where(qempty, nsim, 0.5 * nsim + 0.5 * asim)
comp = comp.with_columns(pl.Series("sc", sc.astype(np.float32)), pl.Series("nsim", nsim.astype(np.float32)),
                         pl.Series("asim", asim.astype(np.float32)),
                         ((pl.col("qn") == pl.col("cn")) & ((pl.col("qa") == pl.col("ca")) & (pl.col("qa") != ""))).alias("exact_na"),
                         (pl.col("qn") == pl.col("cn")).alias("exact_n"))
fnc_ = tgt.select("q", "s1").join(comp, on="q")
parent_sc = fnc_.filter(pl.col("c") == pl.col("s1")).select("q", "s1", pl.col("sc").alias("p_sc"), pl.col("nsim").alias("p_nsim"),
                                                            pl.col("asim").alias("p_asim"), pl.col("exact_na").alias("p_exact_na"),
                                                            pl.col("exact_n").alias("p_exact_n"))
oth = fnc_.filter(pl.col("c") != pl.col("s1")).sort("sc", descending=True).group_by("q", "s1").agg(
    pl.col("sc").first().alias("o_sc"), pl.col("c").first().alias("o_c"), pl.col("exact_na").any().alias("o_exact_na"),
    pl.col("exact_n").any().alias("o_exact_n"), pl.col("exact_n").sum().alias("o_n_exact_n"), pl.len().alias("n_comp"))
fnx = tgt.select("q", "s1", "w", "q_addr_empty").join(parent_sc, on=["q", "s1"], how="left").join(oth, on=["q", "s1"], how="left")
fnx = fnx.with_columns(pl.col("o_sc").fill_null(-1.0), pl.col("o_exact_na").fill_null(False), pl.col("o_exact_n").fill_null(False),
                       pl.col("o_n_exact_n").fill_null(0), pl.col("n_comp").fill_null(0))
fnx = fnx.with_columns((pl.col("p_sc") - pl.col("o_sc")).alias("margin"))
fnx = fnx.with_columns(
    pl.when(pl.col("o_exact_na")).then(pl.lit("irr_exact_collision"))                       # another S1 identical in name+address
      .when(pl.col("q_addr_empty") & pl.col("o_exact_n")).then(pl.lit("irr_empty_addr_namesake"))  # empty address, exact name of another S1
      .when(pl.col("margin") >= 5).then(pl.lit("rec_parent_best_clear"))                   # parent is the unique best match (margin >= 5)
      .when(pl.col("margin") > 0).then(pl.lit("amb_parent_best_narrow"))
      .otherwise(pl.lit("irr_competitor_as_good")).alias("recov"),
    pl.when(pl.col("p_sc") >= 99.5).then(pl.lit("a:=100")).when(pl.col("p_sc") >= 90).then(pl.lit("b:90-99"))
      .when(pl.col("p_sc") >= 80).then(pl.lit("c:80-89")).otherwise(pl.lit("d:<80")).alias("psc_bin"))
fnx.write_parquet(OUT / "recov_all.parquet")
err = err.join(fnx.select("q", "s1", "recov", "psc_bin", "p_sc", "o_sc", "margin", "o_c", "p_nsim", "p_asim", "p_exact_n", "p_exact_na",
                          "n_comp"), on=["q", "s1"], how="left")

# analogue acceptance: among half-B true pairs that reached the scored set (TP + FN_not_chosen + FN_argmax_other + wrong),
# P(chosen | profile) with profile = (address empty, recoverability class, parent-similarity bin). Applied to never-scored FN
# (not retrieved / pruned by Stage A) it estimates how much of their loss wider retrieval / a looser prune would recover.
prof = ["q_addr_empty", "recov", "psc_bin"]
scored_cats = ["TP", "FN_not_chosen", "FN_argmax_other_S1_unassigned", "FN_wrong_S1_assigned"]
ana = fnx.join(t.select("q", "s1", "cat"), on=["q", "s1"]).filter(pl.col("cat").is_in(scored_cats))
acc = ana.group_by(prof).agg(pl.col("w").sum().alias("n_scored_w"),
                             (pl.col("w") * (pl.col("cat") == "TP")).sum().alias("n_tp_w")).with_columns(
    (pl.col("n_tp_w") / pl.col("n_scored_w")).alias("acc"))
err = err.join(acc.select(prof + ["acc"]), on=prof, how="left")
print("\n=== analogue acceptance P(chosen | profile) among scored half-B true pairs ===")
with pl.Config(tbl_rows=60, tbl_width_chars=200):
    print(acc.sort(prof))

# FP records: compare the chosen S1 with the record's real parent (other-S1 copies) / exact look-alikes (orphans)
fpr = err.filter(pl.col("cat").str.starts_with("FP"))
if fpr.height:
    pr_ = fpr.select("q", "s1", "true_s1").join(rec.select(pl.col("rid").alias("q"), pl.col("name_n").alias("qn"), pl.col("addr_n").alias("qa")), on="q") \
             .join(rec.select(pl.col("rid").alias("s1"), pl.col("name_n").alias("sn"), pl.col("addr_n").alias("sa")), on="s1")
    a_, b_, c_, d_ = pr_["qn"].to_list(), pr_["sn"].to_list(), pr_["qa"].to_list(), pr_["sa"].to_list()
    ns_ = 0.5 * cpdist(a_, b_, scorer=fuzz.ratio, workers=2) + 0.5 * cpdist(a_, b_, scorer=fuzz.token_sort_ratio, workers=2)
    as_ = 0.5 * cpdist(c_, d_, scorer=fuzz.token_set_ratio, workers=2) + 0.5 * cpdist(c_, d_, scorer=fuzz.ratio, workers=2)
    fsc = np.where((pr_["qa"] == "").to_numpy(), ns_, 0.5 * ns_ + 0.5 * as_)
    pr_ = pr_.with_columns(pl.Series("fp_sc", fsc.astype(np.float32)),
                           ((pl.col("qn") == pl.col("sn")) & (pl.col("qa") == pl.col("sa"))).alias("fp_exact"))
    err = err.join(pr_.select("q", "s1", "fp_sc", "fp_exact"), on=["q", "s1"], how="left")
err.write_parquet(OUT / "err_records.parquet")

# ------------------------------------------------------------------ slices
def tb(bucket: pl.Expr, name: str, df: pl.DataFrame):
    g = (df.with_columns(bucket.alias(name)).group_by("cat", name).agg(pl.len().alias("n"), pl.col("loss").sum().alias("L"))
           .with_columns((pl.col("L") / NB).alias("F05_units")).sort("F05_units", descending=True))
    piv = g.pivot(on=name, index="cat", values="F05_units", aggregate_function="sum").fill_null(0.0)
    tot = df.with_columns(bucket.alias(name)).group_by(name).agg(pl.len().alias("n"), (pl.col("loss").sum() / NB).alias("F05_units")).sort(name)
    print(f"\n--- slice by {name} (F0.5 units) ---")
    with pl.Config(tbl_rows=40, tbl_cols=20, tbl_width_chars=250, float_precision=6):
        print(tot)
        print(piv)
    rep.setdefault("slices", {})[name] = {str(r[0]): {"n": r[1], "F05": r[2]} for r in tot.iter_rows()}


tb(pl.col("country"), "country", err)
tb(pl.col("q_src").cast(pl.String), "record_source", err)
tb(pl.when(pl.col("T") >= 6).then(pl.lit("6+")).otherwise(pl.col("T").cast(pl.String)), "T_true_copies", err)
tb(pl.col("q_addr_empty").cast(pl.String), "record_addr_empty", err)
tb(pl.when(pl.col("s_name_cnt") <= 1).then(pl.lit("1")).when(pl.col("s_name_cnt") <= 3).then(pl.lit("2-3"))
     .when(pl.col("s_name_cnt") <= 10).then(pl.lit("4-10")).when(pl.col("s_name_cnt") <= 100).then(pl.lit("11-100"))
     .otherwise(pl.lit(">100")), "S1_name_freq", err)
tb(pl.col("recov").fill_null("(FP)"), "recoverability", err)
fnerr = err.filter(pl.col("cat").str.starts_with("FN"))
tb(pl.concat_str([pl.col("recov"), pl.col("q_addr_empty").cast(pl.String)], separator="|"), "recov_x_addrempty", fnerr)
tb(pl.concat_str([pl.col("country"), pl.col("q_addr_empty").cast(pl.String)], separator="|"), "country_x_addrempty", fnerr)
if fpr.height:
    print("\nFP records: exact name+address look-alike of the chosen S1:",
          err.filter(pl.col("cat").str.starts_with("FP")).group_by("cat").agg(pl.len(), pl.col("fp_exact").mean(), pl.col("fp_sc").mean(),
                                                                          pl.col("am_p").mean(), (pl.col("loss").sum() / NB).alias("F05")))

rv = fnerr.group_by("recov").agg(pl.len(), (pl.col("loss").sum() / NB).alias("F05")).sort("F05", descending=True)
print("\n=== FN recoverability (F0.5 units) ===")
print(rv)
rep["fn_recov"] = {r[0]: {"n": r[1], "F05": r[2]} for r in rv.iter_rows()}
cr = fnerr.group_by("cat", "recov").agg(pl.len(), (pl.col("loss").sum() / NB).alias("F05")).sort("F05", descending=True)
with pl.Config(tbl_rows=60):
    print(cr)
rep["cat_x_recov"] = [dict(zip(cr.columns, r)) for r in cr.iter_rows()]

# decision-level diagnostics for FN_not_chosen / wrong-S1
nc = err.filter(pl.col("cat") == "FN_not_chosen")
print("\nFN_not_chosen: argmax prob quantiles", nc["am_p"].quantile(0.1), nc["am_p"].median(), nc["am_p"].quantile(0.9),
      "| S1 got empty prediction:", float(np.mean([not pred.get(s) for s in nc["s1"].to_list()])) if nc.height else None)
never = ["FN_not_retrieved", "FN_pruned_A<0.001", "FN_not_cand", "FN_pruned_A_0.001-0.01"]
nv = err.filter(pl.col("cat").is_in(never))
g = nv.group_by("cat", "q_addr_empty").agg(pl.len().alias("n"), (pl.col("loss").sum() / NB).alias("F05_loss"),
                                           ((pl.col("loss") * pl.col("acc").fill_null(0)).sum() / NB).alias("F05_recoverable_est"))
print("\n=== never-scored FN: loss and analogue-based recoverable estimate (F0.5 units) ===")
with pl.Config(tbl_rows=30):
    print(g.sort("F05_recoverable_est", descending=True))
rep["never_scored_recoverable"] = [dict(zip(g.columns, r)) for r in g.iter_rows()]
g2 = nv.group_by("cat", "recov").agg(pl.len().alias("n"), (pl.col("loss").sum() / NB).alias("F05_loss"),
                                     ((pl.col("loss") * pl.col("acc").fill_null(0)).sum() / NB).alias("F05_recoverable_est"))
with pl.Config(tbl_rows=40):
    print(g2.sort("F05_recoverable_est", descending=True))
(OUT / "e00_report.json").write_text(json.dumps(rep, indent=1, default=str))
print("wrote", OUT / "e00_report.json", OUT / "err_records.parquet")

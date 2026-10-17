"""E00 follow-up 1: post-decision unique-exact-name rule for EMPTY-address S2/S3 records (US/India).

Rule: a record that is unassigned after the production decision (_chosen on val_best_run28_7q, gamma 1.5), has an empty
address, and whose name key equals the key of exactly one pool S1 of its country -> assign it to that S1, respecting the
generator caps (<= 5 S2 and <= 6 S3 copies per S1; existing matches count) and one S1 per record.
Knobs (tuned on half A only): name key strictness (core = name_n without legal words / tokens = all ascii tokens /
raw = lower-cased raw string), minimum number of key tokens, require the S1 to already have >= 1 match, skip records
whose val_best argmax is a different S1. The half-A pick is evaluated once on half B with eval_judge's paired bootstrap
(plain and hard, hard = FP from unmatched records x3).

python scripts/e00_uname_rule.py            # tune on A, report B
python scripts/e00_uname_rule.py --test CFG_JSON   # also write /work/agent_e00/ui_exactname_test.parquet
"""
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "src")
from ber.metric import entity_f05  # noqa: E402
from ber.normalize import name_tokens  # noqa: E402
from ber.pipeline import SEED, hide_pct, hidden_s1, labels  # noqa: E402
from ber.stage2 import _chosen  # noqa: E402

OUT = Path("/work/agent_e00")
work, tag = Path("/work/run28"), "run28_7q"
cfg = {"mode": "efopt", "miss": 0.0, "gamma": 1.5}
CAP = {2: 5, 3: 6}
COUNTRIES = ["US", "India"]
_WS = re.compile(r"\s+")


def keys(names: list[str], kind: str) -> list[str]:
    if kind == "core":
        raise ValueError  # name_n column is used directly
    if kind == "tokens":
        return [" ".join(name_tokens(s)) for s in names]
    return [_WS.sub(" ", (s or "").lower()).strip() for s in names]


def build_keys(rec: pl.DataFrame) -> pl.DataFrame:
    """rec: rid, name_n, business_name -> + key_core, key_tokens, key_raw."""
    return rec.with_columns(pl.col("name_n").alias("key_core"),
                            pl.Series("key_tokens", keys(rec["business_name"].to_list(), "tokens")),
                            pl.Series("key_raw", keys(rec["business_name"].to_list(), "raw")))


def candidates(pool: pl.DataFrame, recs: pl.DataFrame, kind: str) -> pl.DataFrame:
    """Unique-key S1 per (country, key); returns q, s1, src, ntok for records whose key hits exactly one pool S1."""
    k = f"key_{kind}"
    u = pool.filter(pl.col(k) != "").group_by("country", k).agg(pl.len().alias("n"), pl.col("rid").first().alias("s1")) \
            .filter(pl.col("n") == 1).drop("n")
    return recs.filter(pl.col(k) != "").join(u, on=["country", k]).select(
        pl.col("rid").alias("q"), "s1", "src", pl.col(k).str.split(" ").list.len().alias("ntok"))


def apply_rule(c: pl.DataFrame, assigned: set, counts: dict, has_match: set, argmax: dict, min_tok: int, need_match: bool,
               skip_conflict: bool) -> list:
    c = c.filter(pl.col("ntok") >= min_tok).sort(["s1", "src", "q"])
    add, used = [], {}
    for q, s, src in c.select("q", "s1", "src").iter_rows():
        if q in assigned:
            continue
        if need_match and s not in has_match:
            continue
        if skip_conflict and q in argmax and argmax[q] != s:
            continue
        n = counts.get((s, src), 0) + used.get((s, src), 0)
        if n >= CAP[src]:
            continue
        used[(s, src)] = used.get((s, src), 0) + 1
        add.append((q, s))
    return add


def main_val():
    lab = labels(Path("/work/data"), work).with_columns((pl.col("parent").fill_null(pl.col("rid")).hash(SEED + 7) % 2 == 0).alias("half_a"))
    S = {h: lab.filter((pl.col("src") == 1) & pl.col("is_val") & (pl.col("half_a") == (h == "A")))["rid"].to_list() for h in "AB"}
    parent = dict(lab.filter(pl.col("parent").is_not_null()).select("rid", "parent").iter_rows())
    truth = {}
    for q, p in parent.items():
        truth.setdefault(p, set()).add(q)
    orph = set(lab.filter(pl.col("parent").is_null() & (pl.col("src") != 1))["rid"].to_list())
    gt = pl.read_parquet("/work/data/train_pairs.parquet")
    pct = hide_pct(work)
    rec = pl.read_parquet(work / "train" / "records.parquet", columns=["entity_id", "country", "src", "name_n", "addr_empty", "business_name"]
                          ).with_row_index("rid").filter(pl.col("country").is_in(COUNTRIES))
    Hset = set(rec.join(gt.filter(hidden_s1(pl.col("s1"), pct)).select(pl.col("m").alias("entity_id")), on="entity_id", how="semi")["rid"].to_list())
    pool = build_keys(rec.filter((pl.col("src") == 1) & ~hidden_s1(pl.col("entity_id"), pct)))
    er = build_keys(rec.filter((pl.col("src") != 1) & pl.col("addr_empty")))
    print(f"pool S1 {pool.height}, empty-address S2/S3 records {er.height}", flush=True)
    vb = pl.read_parquet(work / f"val_best_{tag}.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
    ch = _chosen(vb, cfg)
    pred = {}
    for q, s in ch.select("q", "s1").iter_rows():
        pred.setdefault(s, set()).add(q)
    assigned = set(ch["q"].to_list())
    src_of = dict(rec.select("rid", "src").iter_rows())
    counts = {}
    for q, s in ch.select("q", "s1").iter_rows():
        counts[(s, src_of.get(q, 0))] = counts.get((s, src_of.get(q, 0)), 0) + 1
    has_match = set(pred)
    argmax = dict(vb.select("q", "s1").iter_rows())

    def fe(s, extra=()):
        p = pred.get(s, set()) | set(extra)
        hard = set(p)
        for q in p:
            if q in orph:
                hard |= {("d", q, 1), ("d", q, 2)}
        t = truth.get(s, ())
        return entity_f05(t, p), entity_f05(t, hard)

    base = {h: {s: fe(s) for s in S[h]} for h in "AB"}
    for h in "AB":
        print(f"base half {h}: plain {np.mean([v[0] for v in base[h].values()]):.6f} hard {np.mean([v[1] for v in base[h].values()]):.6f}")
    cand = {k: candidates(pool, er, k) for k in ("core", "tokens", "raw")}
    grid = list(itertools.product(("core", "tokens", "raw"), (1, 2, 3), (False, True), (False, True)))

    def evaluate(add, h):
        sset = set(S[h])
        by = {}
        for q, s in add:
            if s in sset:
                by.setdefault(s, []).append(q)
        d_plain = np.zeros(len(S[h]))
        d_hard = np.zeros(len(S[h]))
        idx = {s: i for i, s in enumerate(S[h])}
        kinds = {"true": 0, "unmatched_raw": 0, "unmatched_hiddenS1copy": 0, "other_S1_copy": 0}
        for s, qs in by.items():
            a = fe(s, qs)
            d_plain[idx[s]] = a[0] - base[h][s][0]
            d_hard[idx[s]] = a[1] - base[h][s][1]
            for q in qs:
                p = parent.get(q)
                kinds["true" if p == s else "other_S1_copy" if p is not None else "unmatched_hiddenS1copy" if q in Hset else "unmatched_raw"] += 1
        return d_plain, d_hard, len(by), kinds

    rows = []
    for kind, mt, nm, sc in grid:
        add = apply_rule(cand[kind], assigned, counts, has_match, argmax, mt, nm, sc)
        dp, dh, ns, kinds = evaluate(add, "A")
        rows.append({"key": kind, "min_tok": mt, "need_match": nm, "skip_conflict": sc, "A_plain": dp.mean(), "A_hard": dh.mean(),
                     "A_s1": ns, **{f"A_{k}": v for k, v in kinds.items()}})
    g = pl.DataFrame(rows).sort("A_plain", descending=True)
    with pl.Config(tbl_rows=40, tbl_cols=20, tbl_width_chars=250, float_precision=6):
        print("\n=== half A grid (diff vs production) ===")
        print(g)
    ok = g.filter(pl.col("A_hard") >= 0)
    pick = (ok if ok.height else g).row(0, named=True)
    print(f"\nhalf-A pick: {pick}")
    add = apply_rule(cand[pick["key"]], assigned, counts, has_match, argmax, pick["min_tok"], pick["need_match"], pick["skip_conflict"])
    dp, dh, ns, kinds = evaluate(add, "B")
    rng = np.random.default_rng(0)
    res = {"pick": {k: pick[k] for k in ("key", "min_tok", "need_match", "skip_conflict")}, "B_affected_s1": ns, "B_added": kinds}
    for m, d in (("plain", dp), ("hard", dh)):
        n = len(d)
        bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(2000)])
        b0 = np.mean([v[0 if m == "plain" else 1] for v in base["B"].values()])
        res[m] = {"base": b0, "new": b0 + d.mean(), "diff": d.mean(), "ci": [np.quantile(bs, .025), np.quantile(bs, .975)], "p_le0": float(np.mean(bs <= 0))}
        print(f"half B {m}: {b0:.6f} -> {b0 + d.mean():.6f} diff {d.mean():+.6f} CI [{np.quantile(bs, .025):+.6f},{np.quantile(bs, .975):+.6f}] "
              f"P(<=0) {np.mean(bs <= 0):.3f}")
    print(f"half B: affected S1s {ns}, added pairs {kinds}")
    (OUT / "uname_rule_val.json").write_text(json.dumps(res, indent=1, default=float))


def main_test(pick: dict):
    trec = pl.read_parquet(work / "test" / "records.parquet", columns=["entity_id", "country", "src", "name_n", "addr_empty", "business_name"]
                           ).with_row_index("rid")
    sub = pl.read_csv("/work/out_fr13/matching_results.tsv", separator="\t", schema_overrides={"matched_entity_ids": pl.String}) \
            .with_columns(pl.col("matched_entity_ids").fill_null(""))
    eid = trec.select("rid", "entity_id", "src")
    pairs = sub.filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids") \
               .join(eid.rename({"rid": "s1", "entity_id": "source1_entity_id"}).drop("src"), on="source1_entity_id") \
               .join(eid.rename({"rid": "q", "entity_id": "matched_entity_ids"}), on="matched_entity_ids")
    assigned = set(pairs["q"].to_list())
    counts = {}
    for s, src in pairs.select("s1", "src").iter_rows():
        counts[(s, src)] = counts.get((s, src), 0) + 1
    has_match = set(pairs["s1"].to_list())
    tb = pl.read_parquet(work / f"test_best_{tag}.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
    argmax = dict(tb.select("q", "s1").iter_rows())
    rec = trec.filter(pl.col("country").is_in(COUNTRIES))
    pool = build_keys(rec.filter(pl.col("src") == 1))
    er = build_keys(rec.filter((pl.col("src") != 1) & pl.col("addr_empty")))
    c = candidates(pool, er, pick["key"])
    add = apply_rule(c, assigned, counts, has_match, argmax, pick["min_tok"], pick["need_match"], pick["skip_conflict"])
    out = pl.DataFrame(add, schema={"q": pl.UInt32, "s1": pl.UInt32}, orient="row")
    out.write_parquet(OUT / "ui_exactname_test.parquet")
    by_c = out.join(trec.select(pl.col("rid").alias("q"), "country"), on="q").group_by("country").len()
    print(f"test: submission pairs {pairs.height}, assigned records {len(assigned)}; rule adds {out.height} pairs to {out['s1'].n_unique()} S1s; {by_c}")


if __name__ == "__main__":
    if "--test" in sys.argv:
        main_test(json.loads(sys.argv[sys.argv.index("--test") + 1]))
    else:
        main_val()

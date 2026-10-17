"""E01: do French address-normaliser gaps (bis/b, ter/t, cours/crs, quai/q, passage/psg/pass, residence/res,
appartement/app/apt/appt) cost matches?

A pair is a *gap pair* when the S1 and record addresses (France-fixed addr_n tokens, /work/runfr/test/records.parquet)
are unequal as token multisets but equal after mapping the gap abbreviations to one canonical form. House number,
suffix and street stay separate tokens (12, 12 bis, 12 ter never collapse: bis/ter are only matched to b/t, never to
nothing); b/t are read as bis/ter only right after a number.
Control group: France candidate pairs whose addr_n token multisets are already equal (no gap) -> the acceptance a gap pair
should have if the gap did not matter.

Outputs (worker): /work/agent_e01/gap_pairs.parquet (all gap pairs in candidates + retrieval-missed gap pairs, with
context), prints the report tables.
python scripts/e01_gap.py
"""
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import polars as pl
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, "src")
OUT = Path("/work/agent_e01")
OUT.mkdir(exist_ok=True)
pl.Config.set_fmt_str_lengths(70); pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250)

sys.path.insert(0, "/work/agent_e01")
from e01_gap_lib import DECOY, FULL, GAPCLASS, canon, gap_types, name_rel, toks  # noqa: E402

# ------------------------------------------------------------------ records
rec = pl.read_parquet("/work/runfr/test/records.parquet",
                      columns=["entity_id", "src", "country", "business_name", "business_address", "name_n", "addr_n", "house"]) \
        .with_row_index("rid").filter(pl.col("country") == "France").drop("country")
tl = [a.split() for a in rec["addr_n"].to_list()]
key_b = [" ".join(sorted(t)) for t in tl]
key_a = [" ".join(sorted(canon(t))) for t in tl]
has_gap_tok = [any(x in GAPCLASS or x == "quater" for x in t) for t in tl]
rec = rec.with_columns(pl.Series("key_b", key_b), pl.Series("key_a", key_a), pl.Series("gtok", has_gap_tok))
print(f"France records {rec.height}; records whose key changes under canon: {(rec['key_a'] != rec['key_b']).sum()}", flush=True)
s1 = rec.filter(pl.col("src") == 1)
qs = rec.filter(pl.col("src") != 1)

# ------------------------------------------------------------------ decisions
eid = dict(zip(rec["entity_id"].to_list(), rec["rid"].to_list()))


def load_tsv(p):
    rows = []
    for line in open(p):
        a, _, b = line.rstrip("\n").partition("\t")
        if a not in eid or not b or a == "source1_entity_id":
            continue
        for x in b.split(","):
            if x in eid:
                rows.append((eid[x], eid[a]))
    return pl.DataFrame(rows, schema={"q": pl.UInt32, "s1": pl.UInt32}, orient="row")


sub = Path("/work/out_fr12/matching_results.tsv")
if not sub.exists():
    sub = Path("/work/out_fr10/matching_results.tsv")
acc = load_tsv(sub)
print(f"{sub}: France accepted pairs {acc.height}", flush=True)
best7 = pl.read_parquet("/work/run28/test_best_run28_7q.parquet").rename({"s1": "best7", "prob": "p7"})
bestfx = pl.read_parquet("/work/runfr28/test_best_fx7o.parquet").rename({"s1": "bestfx", "prob": "pfx"})
p3 = pl.read_parquet("/work/runfr28/scores_test.parquet")

# ------------------------------------------------------------------ candidate pairs (France-fixed features)
cols = ["q", "s1", "name_ratio", "addr_ratio", "addr_tsr", "street_tsr", "house_eq", "house_miss", "pA"]
cand = pl.concat([pl.read_parquet(p, columns=cols) for p in sorted(Path("/work/runfr/test/pruned").glob("France_*.parquet"))])
cand = cand.with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
print(f"France candidate pairs {cand.height}", flush=True)
R = rec.select("rid", "src", "business_name", "business_address", "addr_n", "house", "key_b", "key_a")
c = cand.join(R.rename({x: x + "_q" for x in R.columns if x != "rid"}), left_on="q", right_on="rid") \
        .join(R.rename({x: x + "_s" for x in R.columns if x != "rid"}), left_on="s1", right_on="rid")
c = c.with_columns((pl.col("key_b_q") == pl.col("key_b_s")).alias("eq_before"),
                   (pl.col("key_a_q") == pl.col("key_a_s")).alias("eq_after"),
                   (pl.col("addr_n_q") == "").alias("q_empty"))
c = c.filter(pl.col("addr_n_q") != "")
print(f"non-empty-address candidate pairs {c.height}: equal before {c['eq_before'].sum()}, equal after {c['eq_after'].sum()}, "
      f"gap pairs (after & not before) {(c['eq_after'] & ~c['eq_before']).sum()}", flush=True)
# also the ordered comparison (segment order kept)
c = c.with_columns(pl.Series("ord_before", [a == b for a, b in zip(c["addr_n_q"].to_list(), c["addr_n_s"].to_list())]))


def context(d):
    d = d.join(acc.with_columns(pl.lit(True).alias("acc")), on=["q", "s1"], how="left").with_columns(pl.col("acc").fill_null(False))
    asg = acc.rename({"s1": "asg_s1"}).group_by("q").agg(pl.col("asg_s1").first())
    d = d.join(asg, on="q", how="left").with_columns(
        pl.when(pl.col("acc")).then(pl.lit("this")).when(pl.col("asg_s1").is_null()).then(pl.lit("unassigned"))
          .otherwise(pl.lit("other")).alias("status"))
    d = d.join(best7, on="q", how="left").join(bestfx, on="q", how="left").join(p3, on=["q", "s1"], how="left")
    cnt = acc.join(rec.select(pl.col("rid").alias("q"), "src"), on="q").group_by("s1").agg(
        (pl.col("src") == 2).sum().alias("n2"), (pl.col("src") == 3).sum().alias("n3"))
    d = d.join(cnt, on="s1", how="left").with_columns(pl.col("n2").fill_null(0), pl.col("n3").fill_null(0))
    d = d.with_columns(
        pl.when(pl.col("acc")).then(pl.col("n2") + pl.col("n3") - 1).otherwise(pl.col("n2") + pl.col("n3")).alias("s1_other_acc"),
        pl.when(pl.col("src_q") == 2).then(pl.col("n2") - pl.col("acc").cast(pl.Int32) < 5)
          .otherwise(pl.col("n3") - pl.col("acc").cast(pl.Int32) < 6).alias("cap_ok"))
    rel = [name_rel(a, b) for a, b in zip(d["business_name_q"].to_list(), d["business_name_s"].to_list())]
    dec = [bool(set(toks(a)) - set(toks(b)) & DECOY) for a, b in zip(d["business_name_q"].to_list(), d["business_name_s"].to_list())]
    return d.with_columns(pl.Series("name_rel", rel), pl.Series("decoy_word", dec),
                          pl.Series("rel_kind", [r.split(":")[0] for r in rel]))


gp = c.filter(pl.col("eq_after") & ~pl.col("eq_before"))
gp = gp.with_columns(pl.Series("gap", [gap_types(a.split(), b.split()) for a, b in zip(gp["key_b_s"].to_list(), gp["key_b_q"].to_list())]))
# direction: S1 has the full form, record the abbreviation (generator direction) vs reverse


def direction(ks, kq):
    ca, cb = Counter(ks.split()), Counter(kq.split())
    s_only, q_only = set((ca - cb).elements()), set((cb - ca).elements())
    if s_only <= FULL and not (q_only & FULL):
        return "s1_full"
    if q_only <= FULL and not (s_only & FULL):
        return "s1_abbr"
    return "mixed"


gp = gp.with_columns(pl.Series("dir", [direction(a, b) for a, b in zip(gp["key_b_s"].to_list(), gp["key_b_q"].to_list())]))
gp = context(gp)
ctrl = context(c.filter(pl.col("eq_before")).sample(n=min(400_000, c.filter(pl.col("eq_before")).height), seed=0))

print("\n=== gap pairs in France candidates, by gap type")
print(gp.group_by("gap").agg(pl.len().alias("pairs"), pl.col("q").n_unique().alias("records"), pl.col("acc").mean().alias("acc_rate"),
                             (pl.col("status") == "unassigned").sum().alias("unassigned"), (pl.col("status") == "other").sum().alias("other_s1"),
                             pl.col("p3").median().alias("p3_med")).sort("pairs", descending=True))
print(gp.group_by("dir").agg(pl.len(), pl.col("acc").mean()))
print("\n=== acceptance by name relation: gap pairs vs control (same address before canon, sample 400k)")
t1 = gp.group_by("rel_kind").agg(pl.len().alias("gap_n"), pl.col("acc").mean().alias("gap_acc"),
                                 (pl.col("status") == "unassigned").mean().alias("gap_unas"), (pl.col("status") == "other").mean().alias("gap_oth"))
t2 = ctrl.group_by("rel_kind").agg(pl.len().alias("ctl_n"), pl.col("acc").mean().alias("ctl_acc"),
                                   (pl.col("status") == "unassigned").mean().alias("ctl_unas"), (pl.col("status") == "other").mean().alias("ctl_oth"))
print(t1.join(t2, on="rel_kind", how="full", coalesce=True).sort("gap_n", descending=True, nulls_last=True))
print("\n=== street_tsr / addr_ratio gap vs control (house_eq rate)")
for nm, d in (("gap", gp), ("ctl", ctrl)):
    print(nm, d.select(pl.col("street_tsr").mean(), pl.col("addr_ratio").mean(), pl.col("addr_tsr").mean(), pl.col("house_eq").mean(), pl.col("p3").mean()))
print("\n=== p3 (fixed features) distribution of gap pairs, exact/case/punct/legal/typo names")
strong = gp.filter(pl.col("rel_kind").is_in(["exact", "case", "punct", "legal", "typo", "reorder"]))
print(strong.select(pl.col("p3").quantile(q).alias(f"q{q}") for q in (0.01, 0.05, 0.1, 0.25, 0.5)))
print(strong.group_by("status").agg(pl.len(), pl.col("p3").median(), pl.col("p7").median()))
cs = ctrl.filter(pl.col("rel_kind").is_in(["exact", "case", "punct", "legal", "typo", "reorder"]))
print("control strong names status:", cs.group_by("status").agg(pl.len(), pl.col("p3").median()))
gp.write_parquet(OUT / "gap_pairs.parquet")
ctrl.write_parquet(OUT / "ctrl_pairs.parquet")

# ------------------------------------------------------------------ retrieval misses: record and S1 with gap-equal address, not a candidate
j = qs.filter(pl.col("gtok") & (pl.col("addr_n") != "")).select(pl.col("rid").alias("q"), "key_a", pl.col("key_b").alias("kbq")) \
      .join(s1.filter(pl.col("gtok")).select(pl.col("rid").alias("s1"), "key_a", pl.col("key_b").alias("kbs")), on="key_a")
j = j.filter(pl.col("kbq") != pl.col("kbs"))
miss = j.join(cand.select("q", "s1"), on=["q", "s1"], how="anti")
print(f"\n=== gap-equal (record, S1) pairs anywhere: {j.height}; not among candidates: {miss.height} "
      f"({miss['q'].n_unique()} records)")
if miss.height:
    m = miss.join(R.rename({x: x + "_q" for x in R.columns if x != "rid"}), left_on="q", right_on="rid") \
            .join(R.rename({x: x + "_s" for x in R.columns if x != "rid"}), left_on="s1", right_on="rid")
    m = context(m.with_columns(pl.lit(None, pl.Float32).alias("pA")))
    print(m.group_by("rel_kind", "status").agg(pl.len()).sort("len", descending=True).head(20))
    print(m.select("business_name_q", "business_name_s", "addr_n_q", "addr_n_s", "status", "p7").head(15))
    m.write_parquet(OUT / "gap_missed.parquet")

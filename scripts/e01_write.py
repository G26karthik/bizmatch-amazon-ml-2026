"""E01 step 4: write the add / conflict / removal lists (no submission files).

adds      (/work/agent_e01/fr_parse_adds.parquet: q, s1, prob, reason): gap pairs (addresses equal only after the corrected
          gap canonicalisation) whose record is UNASSIGNED in fr12, name exact / case / punctuation / legal-form-only /
          token reorder, or one added generic copy word (fils, associes, cie, services), no decoy word, S1 under the per-source
          copy cap, and no other candidate S1 at a gap-equal or exact address with a strong name.
conflicts (fr_parse_conflicts.parquet: q, s1, s1_current, prob, prob_current, reason): the same kind of strong gap pair whose
          record is assigned to a different S1 in fr12; plus accepted suffix-mismatch pairs whose record has a candidate S1
          at the exact address (reassignment suggestion).
removals  (fr_parse_removals.parquet: q, s1, prob, reason): accepted fr12 pairs whose house digits agree but the house
          suffix differs (12 vs 12 bis / 12 bis vs 12 ter), name not strong (copies keep the suffix: 25,519 of 25,525
          strong-name accepted records of suffixed S1s keep it).

python scripts/e01_write.py   (after scripts/e01_gap.py)
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, "/work/agent_e01")
from e01_gap_lib import DECOY, canon, house_suffix, is_initials, name_rel, toks  # noqa: E402

OUT = Path("/work/agent_e01")
pl.Config.set_fmt_str_lengths(50); pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(320)
STRONG = ["exact", "case", "punct", "legal", "reorder"]
GENERIC = {"fils", "associes", "cie", "services"}
SUF = {"bis", "ter", "quater"}
gp = pl.read_parquet(OUT / "gap_pairs.parquet")
rel = gp["name_rel"].to_list()
gp = gp.with_columns(pl.Series("ok_name", [r.split(":")[0] in STRONG or (r.startswith("add1:") and r[5:] in GENERIC) for r in rel]))
# competition: other candidate S1s of the record whose address is gap-equal / exact and whose name is strong
comp = gp.filter(pl.col("ok_name")).group_by("q").agg(pl.col("s1").n_unique().alias("n_ok_s1"))
gp = gp.join(comp, on="q", how="left").with_columns(pl.col("n_ok_s1").fill_null(0))
ctrl_q = pl.read_parquet(OUT / "ctrl_pairs.parquet", columns=["q", "s1", "rel_kind"])  # sample only; exact-address strong competitors
prob = pl.coalesce(pl.when(pl.col("bestfx") == pl.col("s1")).then(pl.col("pfx")), pl.col("p3")).alias("prob")
base = gp.filter(pl.col("ok_name") & ~pl.col("decoy_word") & (pl.col("n_ok_s1") == 1)).with_columns(prob)
adds = base.filter((pl.col("status") == "unassigned") & pl.col("cap_ok"))
adds = adds.with_columns(pl.concat_str(pl.lit("gap:"), pl.col("gap"), pl.lit("|name:"), pl.col("name_rel"), pl.lit("|s1_other_acc:"),
                                       pl.col("s1_other_acc").cast(pl.String)).alias("reason"))
print(f"=== ADD list: {adds.height}")
print(adds.select("q", "s1", "business_name_q", "business_name_s", "addr_n_q", "addr_n_s", "prob", "p3", "p7", "pfx", "best7", "bestfx", "s1_other_acc", "reason"))
adds.select("q", "s1", pl.col("prob").cast(pl.Float32), "reason").write_parquet(OUT / "fr_parse_adds.parquet")

conf = base.filter(pl.col("status") == "other").rename({"asg_s1": "s1_current"})
p3 = pl.read_parquet("/work/runfr28/scores_test.parquet")
conf = conf.join(p3.rename({"s1": "s1_current", "p3": "prob_current"}), on=["q", "s1_current"], how="left")
conf = conf.with_columns(pl.concat_str(pl.lit("gap:"), pl.col("gap"), pl.lit("|name:"), pl.col("name_rel")).alias("reason"))
print(f"\n=== CONFLICT (gap, record assigned elsewhere): {conf.height}")
print(conf.select("q", "s1", "s1_current", "business_name_q", "business_name_s", "addr_n_q", "addr_n_s", "prob", "prob_current", "p7", "best7", "reason"))

# ------------------------------------------------------------------ suffix mismatches among accepted fr12 pairs (any street noise)
rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country", "business_name", "addr_n", "house"]) \
        .with_row_index("rid").filter(pl.col("country") == "France")


def suffix_of(a):
    return house_suffix(a.split())


def rest_key(a):
    """canonical address tokens without the house suffix (token right after the first number)."""
    t = a.split()
    c = canon(t)
    for i, x in enumerate(t):
        if x[:1].isdigit():
            if i + 1 < len(c) and c[i + 1] in SUF:
                c = c[:i + 1] + c[i + 2:]
            break
    return " ".join(sorted(c))


rec = rec.with_columns(pl.Series("suf", [suffix_of(a) for a in rec["addr_n"].to_list()]),
                       pl.Series("rest", [rest_key(a) for a in rec["addr_n"].to_list()]))
eid = dict(zip(rec["entity_id"].to_list(), rec["rid"].to_list()))
rows = []
for line in open("/work/out_fr12/matching_results.tsv"):
    a, _, b = line.rstrip("\n").partition("\t")
    if a in eid and b:
        rows.extend((eid[x], eid[a]) for x in b.split(",") if x in eid)
acc = pl.DataFrame(rows, schema={"q": pl.UInt32, "s1": pl.UInt32}, orient="row")
R = rec.select("rid", "business_name", "addr_n", "house", "suf", "rest")
x = acc.join(R.rename({c: c + "_q" for c in R.columns if c != "rid"}), left_on="q", right_on="rid") \
       .join(R.rename({c: c + "_s" for c in R.columns if c != "rid"}), left_on="s1", right_on="rid")
x = x.filter((pl.col("house_q") == pl.col("house_s")) & (pl.col("house_q") != "") & (pl.col("suf_q") != pl.col("suf_s")))
x = x.with_columns(pl.Series("rel", [name_rel(a, b) for a, b in zip(x["business_name_q"].to_list(), x["business_name_s"].to_list())]))
x = x.with_columns(pl.col("rel").str.split(":").list.first().is_in(STRONG).alias("strong"),
                   pl.Series("initials", [is_initials(a, b) for a, b in zip(x["business_name_q"].to_list(), x["business_name_s"].to_list())]))      .join(p3, on=["q", "s1"], how="left")
print(f"\n=== accepted fr12 France pairs with equal house digits but different suffix: {x.height} (strong names {x['strong'].sum()})")
print(x.group_by("suf_s", "suf_q", "strong").len().sort("len", descending=True))
print(x.select("q", "s1", "business_name_q", "business_name_s", "addr_n_q", "addr_n_s", "rel", "p3").head(40))
print("initials-name suffix mismatches (kept, copies):", x.filter(pl.col("initials")).select("business_name_q", "business_name_s", "addr_n_q", "addr_n_s"))
# removal: weak (unrelated / pseudo) name, not initials, not a typo or non-word scramble of the S1 name, and the rest of
# the address identical (a clean suffix difference, not a typo inside the suffix / a dropped house number)
x = x.with_columns(pl.col("rel").str.starts_with("typo").alias("typo_name"), (pl.col("rest_q") == pl.col("rest_s")).alias("clean"))
print("suffix mismatches: clean", x["clean"].sum(), "| strong", x["strong"].sum(), "| initials", x["initials"].sum(), "| typo", x["typo_name"].sum())
rm = x.filter(~pl.col("strong") & ~pl.col("initials") & ~pl.col("typo_name") & pl.col("clean")
              & ~pl.col("rel").str.starts_with("swap1")).with_columns(pl.concat_str(pl.lit("suffix:"), pl.col("suf_s"), pl.lit(">"), pl.col("suf_q"), pl.lit("|name:"),
                                                          pl.col("rel")).alias("reason"))
rm.select("q", "s1", pl.col("p3").alias("prob"), "reason").write_parquet(OUT / "fr_parse_removals.parquet")
print(f"removal list: {rm.height}")

# reassignment suggestions: the removed record has a France candidate S1 at the exact address with the SAME suffix
cand = pl.concat([pl.read_parquet(p, columns=["q", "s1"]) for p in sorted(Path("/work/runfr/test/pruned").glob("France_*.parquet"))]) \
         .with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
key = rec.select("rid", pl.Series("k", [" ".join(sorted(canon(a.split()))) for a in rec["addr_n"].to_list()]))
alt = rm.select("q", pl.col("s1").alias("s1_current"), pl.col("p3").alias("prob_current")).join(cand, on="q") \
        .join(key.rename({"rid": "q", "k": "kq"}), on="q").join(key.rename({"rid": "s1", "k": "ks"}), on="s1") \
        .filter(pl.col("kq") == pl.col("ks")).join(p3, on=["q", "s1"], how="left").rename({"p3": "prob"}) \
        .join(rec.select(pl.col("rid").alias("q"), pl.col("business_name").alias("nq")), on="q")         .join(rec.select(pl.col("rid").alias("s1"), pl.col("business_name").alias("ns")), on="s1")
alt = alt.with_columns(pl.Series("ini", [is_initials(a, b) for a, b in zip(alt["nq"].to_list(), alt["ns"].to_list())]))          .with_columns(pl.when(pl.col("ini")).then(pl.lit("suffix_mismatch_reassign_to_exact_address|initials_match"))
                         .otherwise(pl.lit("suffix_mismatch_reassign_to_exact_address|pseudo_name_low_conf")).alias("reason"))
print(f"reassignment suggestions (exact-address S1 for removed records): {alt.height}")
print(alt.head(20))
out_conf = pl.concat([conf.select("q", "s1", "s1_current", pl.col("prob").cast(pl.Float32), pl.col("prob_current").cast(pl.Float32), "reason"),
                      alt.select("q", "s1", "s1_current", pl.col("prob").cast(pl.Float32), pl.col("prob_current").cast(pl.Float32), "reason")])
out_conf.write_parquet(OUT / "fr_parse_conflicts.parquet")
print(f"wrote adds {adds.height}, conflicts {out_conf.height}, removals {rm.height}")

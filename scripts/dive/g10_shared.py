"""Generator dive 10 (worker): shared noise between copies of one S1 (rare novel tokens / exact same garble), and
S1 decoy-count x copy-count structure. Uses groups_sample.parquet (g4) and s1_counts.parquet."""
import itertools
import re

import polars as pl

pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(220)
cc = pl.read_parquet("/work/dive/g4_cc.parquet")
print(cc.group_by("country", "pair").agg(pl.len(), pl.col("n_eq").mean().round(4), pl.col("n_eq").filter(pl.col("n_both_noisy")).mean().round(4).alias("n_eq|both_noisy"),
      (pl.col("n_cc") > pl.col("n_cs")).filter(pl.col("n_both_noisy")).mean().round(4).alias("n_closer|noisy"),
      pl.col("a_eq").filter(pl.col("a_both_nonempty") & pl.col("a_both_noisy")).mean().round(4).alias("a_eq|both_noisy")).sort("country", "pair"))
g = pl.read_parquet("/work/dive/groups_sample.parquet")
TOK = re.compile(r"[a-z0-9]+")
COMMON = {"center", "services", "service", "partners", "com", "the", "a", "llc", "inc", "ltd", "limited", "private", "pvt", "co",
          "corp", "corporation", "company", "group", "holdings", "and", "of", "www", "formerly", "dba", "aka", "fka", "nee",
          "doing", "business", "as", "known", "also", "trading", "labs", "sys", "one", "id", "incorporated", "lp", "llp", "pc", "pllc"}
rows = []
for (s1r,), grp in g.group_by(["s1r"]):
    S = grp.filter(pl.col("kind") == "s").to_dicts()
    if not S:
        continue
    S = S[0]
    st = set(TOK.findall((S["business_name"] or "").lower()))
    sa = set(TOK.findall((S["business_address"] or "").lower()))
    C = grp.filter(pl.col("kind") == "c").to_dicts()
    for a, b in itertools.combinations(C, 2):
        ta = set(TOK.findall((a["business_name"] or "").lower())) - st - COMMON
        tb = set(TOK.findall((b["business_name"] or "").lower())) - st - COMMON
        ta = {t for t in ta if len(t) >= 4 and not t.isdigit()}; tb = {t for t in tb if len(t) >= 4 and not t.isdigit()}
        aa = {t for t in set(TOK.findall((a["business_address"] or "").lower())) - sa if len(t) >= 4 and not t.isdigit()}
        ab = {t for t in set(TOK.findall((b["business_address"] or "").lower())) - sa if len(t) >= 4 and not t.isdigit()}
        rows.append(dict(country=S["country"], pair="".join(sorted([f"S{a['src']}", f"S{b['src']}"])),
                         a_novel=len(ta) > 0, b_novel=len(tb) > 0, shared_novel=len(ta & tb) > 0,
                         a_anovel=len(aa) > 0, b_anovel=len(ab) > 0, shared_anovel=len(aa & ab) > 0))
r = pl.DataFrame(rows)
print("\nnovel (not in S1, len>=4, not a common add-on word) name / address tokens shared by two copies of one S1:")
print(r.group_by("country", "pair").agg(pl.len(),
      (pl.col("a_novel") & pl.col("b_novel")).mean().round(4).alias("both_have_novel_name_tok"),
      pl.col("shared_novel").filter(pl.col("a_novel") & pl.col("b_novel")).mean().round(4).alias("shared|both_novel"),
      (pl.col("a_anovel") & pl.col("b_anovel")).mean().round(4).alias("both_novel_addr_tok"),
      pl.col("shared_anovel").filter(pl.col("a_anovel") & pl.col("b_anovel")).mean().round(4).alias("addr_shared|both")).sort("country", "pair"))
s = pl.read_parquet("/work/dive/s1_counts.parquet").with_columns((pl.col("n2") + pl.col("n3")).alias("ncop"))
print("\nS1 decoy count (any candidate strength) by whether the S1 has copies:")
print(s.group_by("country", (pl.col("ncop") > 0).alias("has_copy")).agg(pl.len(), (pl.col("nd_any") == 0).mean().round(4).alias("p0"),
      (pl.col("nd_any") == 1).mean().round(4).alias("p1"), (pl.col("nd_any") >= 2).mean().round(4).alias("p2+"), pl.col("nd_any").mean().round(3)).sort("country", "has_copy"))

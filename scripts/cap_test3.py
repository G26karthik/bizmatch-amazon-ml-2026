"""Cap test on more France record classes (best-S1 / accepted pairs), plus the same classes in US as calibration:
  I_exact     identical name key, exact address (copy reference)
  I_hmiss     identical name key, same street words, one side has no house number
  I_hdiff     identical name key, same street words, both houses, different, not a decoy offset
  I_street    identical name key, same house, street words differ (edit)
  I_far       identical name key, house and street differ
  I_empty     identical name key, record address empty
  P_exact     no shared word (pseudo / brand / web / initials), exact address, 1 S1 at the address
  P_multi     same but 2+ S1s at the address
  L_exact     legal-only change, exact address (copy reference)
Rate per 1k S1 by other accepted records of the S1 in the same source; copies vanish at the cap (S2 5, S3 6).

python scripts/cap_test3.py /work/out_fr10
"""
import re
import sys
import unicodedata

import polars as pl

out = sys.argv[1]
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(250)
LF = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "inc", "llc", "ltd", "corp", "corporation", "incorporated", "co", "company",
      "lp", "llp", "pllc", "pc", "plc", "pvt", "private", "limited"}
STOP = {"de", "des", "du", "la", "le", "les", "et", "d", "l", "s", "a", "the", "of", "and", "com", "www"}
OFFS = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country", "business_name", "addr_n", "house"]) \
        .with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
names = rec["business_name"].to_list(); addrs = rec["addr_n"].to_list(); hs = rec["house"].to_list(); srcl = rec["src"].to_list(); ctry = rec["country"].to_list()
tok = [set(re.findall(r"[a-z0-9]+", norm(n).replace(".", ""))) for n in names]
nk = [frozenset(t - LF - STOP) for t in tok]
lf = [frozenset(t & LF) for t in tok]
at = [frozenset(re.findall(r"[a-z0-9]+", norm(a))) if a else frozenset() for a in addrs]
sw = [frozenset(x for x in t if not x.isdigit()) for t in at]
eid = rec.select("rid", "entity_id")
g = pl.read_csv(f"{out}/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls().filter(pl.col("matched_entity_ids") != "") \
       .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
       .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1")
tb = pl.read_parquet("/work/run28/test_best_run28_7q.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32)).select("q", "s1")
pool = pl.concat([tb, acc]).unique()
s1_at_addr = {}
for i in range(len(srcl)):
    if srcl[i] == 1 and at[i]:
        s1_at_addr[(ctry[i], at[i])] = s1_at_addr.get((ctry[i], at[i]), 0) + 1


def cat(q, s):
    if not at[q]:
        return "I_empty" if nk[q] == nk[s] and nk[q] else None
    same_addr = at[q] == at[s]
    if nk[q] == nk[s] and nk[q]:
        if same_addr:
            return "I_exact" if lf[q] == lf[s] else "L_exact"
        if sw[q] == sw[s]:
            if not hs[q] or not hs[s]:
                return "I_hmiss"
            if hs[q].isdigit() and hs[s].isdigit() and int(hs[q]) - int(hs[s]) in OFFS:
                return None
            return "I_hdiff"
        if hs[q] and hs[q] == hs[s]:
            return "I_street"
        return "I_far"
    if same_addr and not (nk[q] & nk[s]):
        return "P_exact" if s1_at_addr.get((ctry[s], at[s]), 0) == 1 else "P_multi"
    return None


pool = pool.with_columns(pl.Series("cat", [cat(q, s) for q, s in zip(pool["q"].to_list(), pool["s1"].to_list())])).filter(pl.col("cat").is_not_null())
pool = pool.join(acc.with_columns(pl.lit(True).alias("acc")), on=["q", "s1"], how="left").with_columns(pl.col("acc").fill_null(False))
pool = pool.with_columns(pl.Series("src", [srcl[q] for q in pool["q"].to_list()], dtype=pl.Int8), pl.Series("country", [ctry[s] for s in pool["s1"].to_list()]))
accs = acc.with_columns(pl.Series("src", [srcl[q] for q in acc["q"].to_list()], dtype=pl.Int8))
cnt = accs.group_by("s1", "src").agg(pl.len().alias("n_acc"))
x = pool.join(cnt, on=["s1", "src"], how="left").with_columns(pl.col("n_acc").fill_null(0)).with_columns((pl.col("n_acc") - pl.col("acc").cast(pl.Int32)).alias("n_other"))
s1f = rec.filter(pl.col("src") == 1).select(pl.col("rid").alias("s1"), "country")
base = pl.concat([s1f.with_columns(pl.lit(s, dtype=pl.Int8).alias("src")) for s in (2, 3)]).join(cnt, on=["s1", "src"], how="left") \
         .with_columns(pl.col("n_acc").fill_null(0))
for country in ("US", "France"):
    b = base.filter(pl.col("country") == country)
    # denominator per n_other level: S1-source slots with n_acc = n (for rejected) — use n_acc level
    den = b.group_by("src", "n_acc").agg(pl.len().alias("n_s1")).rename({"n_acc": "n_other"})
    t = x.filter(pl.col("country") == country).group_by("src", "n_other", "cat").agg(pl.len().alias("n")).join(den, on=["src", "n_other"]) \
         .with_columns((pl.col("n") / pl.col("n_s1") * 1000).round(2).alias("per1k"))
    print(f"\n=== {country}: per 1k S1 slots by other accepted records in the source")
    print(t.filter(pl.col("src") == 3).pivot(on="cat", index="n_other", values="per1k").fill_null(0).sort("n_other"))
    print(t.filter(pl.col("src") == 2).pivot(on="cat", index="n_other", values="per1k").fill_null(0).sort("n_other"))
    print("counts/accepted:", x.filter(pl.col("country") == country).group_by("cat").agg(pl.len(), pl.col("acc").sum()).sort("cat").rows())
x.filter(pl.col("country") == "France").select("q", "s1", "cat", "acc", "src", "n_other").write_parquet("/work/dive/fr_cap3.parquet")

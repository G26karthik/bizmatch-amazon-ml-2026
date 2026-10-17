"""Missed-copy search, model-free. France records that NO S1 claims (unassigned in the output), paired structurally with an
S1: same street (house removed) and name relation (same name key / one-word swap / word overlap), house relation
(same house, decoy offset, other). For each class, rate per 1k S1-source slots by the S1's accepted records in the
record's source; copies vanish at the cap (S2 5, S3 6), decoys and namesakes do not. Classes that vanish = missed copies.
Also US for calibration (its unassigned set is validated as mostly non-copies).

python scripts/unassigned_cap.py /work/out_fr11
"""
import re
import sys
import unicodedata

import polars as pl

out = sys.argv[1]
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(230)
LEG = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "inc", "llc", "ltd", "pvt", "private", "limited", "corp", "corporation",
       "incorporated", "company", "co", "lp", "llp", "pllc", "pc", "plc", "the", "de", "des", "du", "la", "le", "les", "et", "of", "and", "s", "a", "d", "l"}
DW = {"holding", "participations", "distribution", "international", "groupe", "developpement", "snc", "southside", "eastgate", "northside",
      "midtown", "greater"}
OFFS = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country", "business_name", "addr_n", "house"]) \
        .with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
names = rec["business_name"].to_list(); addrs = rec["addr_n"].to_list(); hs = rec["house"].to_list(); srcl = rec["src"].to_list(); ctry = rec["country"].to_list()
nt = [set(re.findall(r"[a-z0-9]+", norm(n))) - LEG for n in names]
raw_nt = [set(re.findall(r"[a-z0-9]+", norm(n))) for n in names]
at = [set(re.findall(r"[a-z0-9]+", norm(a))) if a else set() for a in addrs]
street = [" ".join(sorted(t - {h})) if h else " ".join(sorted(t)) for t, h in zip(at, hs)]
hn = [int(h) if h and h.isdigit() else None for h in hs]
eid = rec.select("rid", "entity_id")
g = pl.read_csv(f"{out}/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls().filter(pl.col("matched_entity_ids") != "") \
       .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
       .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1")
assigned = set(acc["q"].to_list())
accs = acc.with_columns(pl.Series("src", [srcl[q] for q in acc["q"].to_list()], dtype=pl.Int8))
cnt = {(s, sr): n for s, sr, n in accs.group_by("s1", "src").len().iter_rows()}
rows = []
for country in ("US", "France"):
    by_street = {}
    for i in range(len(srcl)):
        if ctry[i] == country and srcl[i] == 1 and street[i]:
            by_street.setdefault(street[i], []).append(i)
    for i in range(len(srcl)):
        if ctry[i] != country or srcl[i] == 1 or i in assigned or not street[i]:
            continue
        if raw_nt[i] & DW:
            continue
        best = None
        for s in by_street.get(street[i], []):
            A, B = nt[i], nt[s]
            if not A or not B:
                continue
            if A == B:
                rel = "same_name"
            elif len(A - B) == 1 and len(B - A) == 1 and A & B:
                rel = "swap1"
            elif len(A ^ B) == 1:
                rel = "add_drop"
            elif A & B:
                rel = "overlap"
            else:
                continue
            if hn[i] is not None and hn[s] is not None:
                d = hn[i] - hn[s]
                hrel = "same_house" if d == 0 else ("offset" if d in OFFS else "house_other")
            else:
                hrel = "no_house"
            rank = ("same_name", "add_drop", "swap1", "overlap").index(rel)
            if best is None or rank < best[0]:
                best = (rank, s, rel, hrel)
        if best is None:
            continue
        _, s, rel, hrel = best
        rows.append((country, i, s, f"{rel}|{hrel}", srcl[i], cnt.get((s, srcl[i]), 0)))
x = pl.DataFrame(rows, schema=["country", "q", "s1", "cls", "src", "n_other"], orient="row")
s1s = rec.filter(pl.col("src") == 1).select(pl.col("rid").alias("s1"), "country")
slots = pl.concat([s1s.with_columns(pl.lit(s, dtype=pl.Int64).alias("src")) for s in (2, 3)])
slots = slots.with_columns(pl.Series("n_other", [cnt.get((s, sr), 0) for s, sr in zip(slots["s1"].to_list(), slots["src"].to_list())]))
den = slots.group_by("country", "src", "n_other").agg(pl.len().alias("n_slots"))
t = x.group_by("country", "cls", "src", "n_other").agg(pl.len().alias("n")).join(den, on=["country", "src", "n_other"])
for country in ("US", "France"):
    y = t.filter(pl.col("country") == country)
    tot = x.filter(pl.col("country") == country).group_by("cls").agg(pl.len().alias("unassigned"))
    lo = y.filter(pl.col("n_other").is_between(2, 3)).group_by("cls").agg((pl.col("n").sum() / pl.col("n_slots").sum() * 1000).alias("rate_n23"))
    cap = y.filter(((pl.col("src") == 2) & (pl.col("n_other") >= 5)) | ((pl.col("src") == 3) & (pl.col("n_other") >= 6))) \
           .group_by("cls").agg((pl.col("n").sum() / pl.col("n_slots").sum() * 1000).alias("rate_cap"))
    z = tot.join(lo, on="cls", how="left").join(cap, on="cls", how="left").fill_null(0).with_columns((pl.col("rate_cap") / pl.col("rate_n23")).round(3).alias("cap_vs_n23"))
    print(f"\n=== {country}: unassigned records paired structurally with an S1 on the same street")
    print(z.sort("unassigned", descending=True).with_columns([pl.col(c).round(2) for c in ("rate_n23", "rate_cap")]).head(20))
x.filter(pl.col("country") == "France").write_parquet("/work/dive/fr_unassigned_struct.parquet")

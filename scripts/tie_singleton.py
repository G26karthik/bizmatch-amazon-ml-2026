"""Namesake ties for empty-address records, broken by the singleton prior. Train truth (US/India, no hiding): an
empty-address S2/S3 record whose name key (legal/stop removed) matches 2+ S1s; "found" copies of each namesake =
its true copies WITH an address (the pipeline finds ~99% of those; proxy for production). Rule: exactly one namesake
has >= 1 found copy -> the record goes to it. Precision (= the record's true parent), coverage, and the same with
"found" >= 2. Then counts on test with fr14 as "found" (namesakes with >= 1 accepted match), for unassigned records.

uv run --no-project --with polars==1.44.2 python scripts/tie_singleton.py
"""
import re
import unicodedata

import polars as pl

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(220)
CANON = {"ltd", "limited", "pvt", "private", "inc", "incorporated", "corp", "corporation", "co", "company", "llc", "lp", "llp", "pc",
         "pllc", "plc", "sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei"}
STOP = {"the", "of", "and", "de", "des", "du", "la", "le", "les", "et", "s", "a", "d", "l"}


def key(s):
    t = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower().replace(".", "")
    return " ".join(sorted(set(re.findall(r"[a-z0-9]+", t)) - CANON - STOP))


def empty(col):
    return pl.col(col).is_null() | (pl.col(col).str.strip_chars() == "")


s1 = pl.read_parquet("local_data/train_source1.parquet").select("entity_id", "business_name", "country")
oth = pl.concat([pl.read_parquet(f"local_data/train_source{s}.parquet") for s in (2, 3)], how="vertical_relaxed").select("entity_id", "business_name",
                                                                                                                          "business_address", "country")
gt = pl.read_parquet("local_data/train_pairs.parquet")
oth = oth.join(gt.rename({"m": "entity_id", "s1": "parent"}), on="entity_id", how="left").with_columns(empty("business_address").alias("empty"))
found = oth.filter(~pl.col("empty") & pl.col("parent").is_not_null()).group_by("parent").len().rename({"parent": "entity_id", "len": "found"})
s1 = s1.join(found, on="entity_id", how="left").with_columns(pl.col("found").fill_null(0), pl.col("business_name").map_elements(key, return_dtype=pl.String).alias("k"))
grp = s1.group_by("country", "k").agg(pl.col("entity_id").alias("cands"), pl.col("found").alias("fnd"), pl.len().alias("n")).filter(pl.col("n") >= 2)
e = oth.filter(pl.col("empty")).with_columns(pl.col("business_name").map_elements(key, return_dtype=pl.String).alias("k")).join(grp, on=["country", "k"])
rows = []
for par, cands, fnd, n, c in e.select("parent", "cands", "fnd", "n", "country").iter_rows():
    for thr in (1, 2):
        nz = [cc for cc, f in zip(cands, fnd) if f >= thr]
        pick = nz[0] if len(nz) == 1 else None
        rows.append((c, thr, n, pick is not None, pick == par if pick else None, par is None))
x = pl.DataFrame(rows, schema=["country", "thr", "n", "fires", "ok", "orphan"], orient="row")
print("train empty-address records in 2+ namesake groups:", e.height)
print(x.group_by("country", "thr").agg(pl.len().alias("records"), pl.col("fires").sum().alias("fires"), pl.col("ok").mean().round(4).alias("precision"),
      (pl.col("fires") & pl.col("orphan")).sum().alias("fires_on_orphans")).sort("country", "thr"))
print(x.filter(pl.col("fires") & (pl.col("thr") == 1)).group_by("country", pl.col("n").clip(2, 6)).agg(pl.len(), pl.col("ok").mean().round(4)).sort("country", "n"))

# ---- test: fr14 accepted matches as "found"
t1 = pl.read_parquet("local_data/test_source1.parquet").select("entity_id", "business_name", "country")
t2 = pl.concat([pl.read_parquet(f"local_data/test_source{s}.parquet") for s in (2, 3)], how="vertical_relaxed").select("entity_id", "business_name",
                                                                                                                         "business_address", "country")
g = pl.read_csv("output/v28q_fr15/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = g.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
nacc = acc.group_by("source1_entity_id").len().rename({"source1_entity_id": "entity_id", "len": "found"})
assigned = set(acc["matched_entity_ids"].to_list())
t1 = t1.join(nacc, on="entity_id", how="left").with_columns(pl.col("found").fill_null(0), pl.col("business_name").map_elements(key, return_dtype=pl.String).alias("k"))
tg = t1.group_by("country", "k").agg(pl.col("entity_id").alias("cands"), pl.col("found").alias("fnd"), pl.len().alias("n")).filter(pl.col("n") >= 2)
te = t2.filter(empty("business_address") & ~pl.col("entity_id").is_in(list(assigned))).with_columns(
    pl.col("business_name").map_elements(key, return_dtype=pl.String).alias("k")).join(tg, on=["country", "k"])
out = []
for eid, cands, fnd, n, c in te.select("entity_id", "cands", "fnd", "n", "country").iter_rows():
    nz = [cc for cc, f in zip(cands, fnd) if f >= 1]
    if len(nz) == 1:
        out.append((eid, nz[0], c, n))
y = pl.DataFrame(out, schema=["m", "s1", "country", "n"], orient="row")
print("\ntest unassigned empty-address records in 2+ namesake groups:", te.height, " rule fires:", y.height)
print(y.group_by("country").len().sort("country"))
y.write_parquet("local_data/work/tie_singleton_test.parquet")

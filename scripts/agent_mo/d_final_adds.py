"""agent_mo final: write adds for the classes that pass (train P >= 0.85, no decoy excess, implied residual precision OK):
India empty-address two-op 'add+add' and 'add+typo' (task A). US A classes and all task-B classes fail the excess check.
Kind carries a 'decor' flag: record-only words are all decorations (id / digits / honorifics / typo of a legal word).
Caps (<= 5 S2 / <= 6 S3 per S1, counting fr27's matches) enforced.

uv run --no-project --with polars==1.44.2 --with rapidfuzz python scripts/agent_mo/d_final_adds.py
"""
import re
import unicodedata

import polars as pl
from rapidfuzz.distance import Levenshtein

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(60)
LEG = {"inc", "llc", "ltd", "limited", "private", "pvt", "corp", "corporation", "co", "company", "incorporated", "lp", "llp",
       "pc", "plc", "the", "and", "of", "l", "c", "p", "d", "s", "a", "sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "de", "du", "des", "la", "le", "les", "et"}
HON = {"id", "sri", "shri", "smt", "mr", "mrs", "ms", "dr", "m", "shree"}
PREC = {"add+add": 0.990, "add+typo": 0.98}          # train (all visible) India: 405/409, 51/51


def toks(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    return re.findall(r"\w+", "".join(ch for ch in s if not unicodedata.combining(ch)))


def decor(w):
    return w in HON or w.isdigit() or any(len(l) >= 5 and Levenshtein.distance(w, l) <= 2 for l in ("private", "limited", "company", "corporation"))


te = pl.read_parquet("local_data/work/agent_mo_A_test_India.parquet")
print(te.filter(pl.col("op").is_in(list(PREC))).group_by("op", "status").len().sort("op", "status"))
p = te.filter((pl.col("status") == "unassigned") & pl.col("op").is_in(list(PREC)))
ids = pl.concat([p["m"], p["s1"]]).implode()
nm = pl.concat([pl.scan_parquet(f"local_data/test_source{s}.parquet").select("entity_id", "business_name") for s in (1, 2, 3)]).filter(pl.col("entity_id").is_in(ids)).collect()
nm = dict(zip(nm["entity_id"], nm["business_name"]))
rows = []
for r in p.iter_rows(named=True):
    R, T = set(toks(nm[r["m"]])) - LEG, set(toks(nm[r["s1"]])) - LEG
    d = all(decor(w) for w in R - T)
    rows.append((r["m"], r["s1"], "India", f"A_empty_twoop:{r['op']}:{'decor' if d else 'other'}", PREC[r["op"]], nm[r["m"]], nm[r["s1"]]))
p = pl.DataFrame(rows, schema=["m", "s1", "country", "kind", "prec", "name_m", "name_s1"], orient="row")
g = pl.read_csv("output/v28q_fr27/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = g.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "m"})
assert p.join(acc, on="m", how="semi").height == 0
have = acc.join(p.select("s1").unique(), on="s1", how="semi").group_by("s1", pl.col("m").str.slice(0, 2).alias("src")).len().rename({"len": "have"})
p = p.with_columns(pl.col("m").str.slice(0, 2).alias("src")).join(have, on=["s1", "src"], how="left").with_columns(pl.col("have").fill_null(0))
p = p.with_columns((pl.col("have") + pl.int_range(1, pl.len() + 1).over("s1", "src")).alias("tot"))
print("dropped by caps:", p.filter(pl.col("tot") > pl.when(pl.col("src") == "S2").then(5).otherwise(6)).height)
p = p.filter(pl.col("tot") <= pl.when(pl.col("src") == "S2").then(5).otherwise(6))
print(p.select("kind", "have", "name_m", "name_s1").sort("kind"))
p.select("m", "s1", "country", "kind", "prec").write_parquet("local_data/work/agent_mo_adds.parquet")
print(p.group_by("kind").len().sort("kind"))

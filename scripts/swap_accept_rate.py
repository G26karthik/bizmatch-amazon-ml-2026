"""Acceptance rate of one-word-swap records at the S1's exact address (unique S1 there), per country, in a test output.
Train truth: US 96.1% / India 97.8% of such records are true copies (swap_cluster_test). US test F equals validation, so
the US acceptance rate is the calibrated reference; a much lower France rate means missed copies (FN).
Breaks down by: swapped-in word seen in any S1 name (type word) or novel; legal form kept; accepted elsewhere.

python scripts/swap_accept_rate.py /work/out_fr7 [--write /work/dive/swap_unassigned_fr.parquet]
"""
import re
import sys
import unicodedata

import polars as pl

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_fmt_str_lengths(50); pl.Config.set_tbl_width_chars(250)
out = sys.argv[1]
write = sys.argv[sys.argv.index("--write") + 1] if "--write" in sys.argv else None
LEGAL = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "inc", "llc", "ltd", "pvt", "private", "limited", "corp", "corporation",
         "incorporated", "company", "co", "lp", "llp", "pllc", "pc", "plc", "the", "de", "des", "du", "la", "le", "les", "et", "of", "and", "s", "a", "d", "l"}


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country", "business_name", "addr_n", "house"]).with_row_index("rid") \
        .with_columns(pl.col("rid").cast(pl.UInt32))
rec = rec.with_columns(pl.Series("ak", [" ".join(sorted(set(re.findall(r"[a-z0-9]+", norm(a))))) if a and h else "" for a, h in zip(rec["addr_n"].to_list(), rec["house"].to_list())]),
                       pl.Series("nt", [sorted(set(re.findall(r"[a-z0-9]+", norm(n))) - LEGAL) for n in rec["business_name"].to_list()]))
eid = rec.select("rid", "entity_id")
g = pl.read_csv(f"{out}/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls("matched_entity_ids") \
       .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1a"}), on="source1_entity_id") \
       .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1a")
s1 = rec.filter((pl.col("src") == 1) & (pl.col("ak") != "")).select(pl.col("rid").alias("s1"), "country", "ak", pl.col("nt").alias("nts"))
s1 = s1.join(s1.group_by("country", "ak").len(), on=["country", "ak"]).filter(pl.col("len") == 1).drop("len")
s1_vocab = {c: set(w for ws in s1.filter(pl.col("country") == c)["nts"].to_list() for w in ws) for c in ("US", "India", "France")}
q = rec.filter((pl.col("src") != 1) & (pl.col("ak") != "")).select(pl.col("rid").alias("q"), "country", "ak", pl.col("nt").alias("ntq"))
d = q.join(s1, on=["country", "ak"])
sw, out_w = [], []
for a, b in zip(d["ntq"].to_list(), d["nts"].to_list()):
    A, B = set(a), set(b)
    ok = len(A - B) == 1 and len(B - A) == 1 and bool(A & B)
    sw.append(next(iter(A - B)) if ok else None)
    out_w.append(next(iter(B - A)) if ok else None)
d = d.with_columns(pl.Series("sw_in", sw), pl.Series("sw_out", out_w)).filter(pl.col("sw_in").is_not_null())
d = d.with_columns(pl.Series("in_vocab", [w in s1_vocab[c] for w, c in zip(d["sw_in"].to_list(), d["country"].to_list())]))
d = d.join(acc, on="q", how="left").with_columns(
    pl.when(pl.col("s1a").is_null()).then(pl.lit("unassigned")).when(pl.col("s1a") == pl.col("s1")).then(pl.lit("accepted")).otherwise(pl.lit("elsewhere")).alias("state"))
print(d.group_by("country", "in_vocab").agg(pl.len().alias("n"), (pl.col("state") == "accepted").mean().round(3).alias("acc_rate"),
                                            (pl.col("state") == "unassigned").mean().round(3).alias("unassigned"),
                                            (pl.col("state") == "elsewhere").mean().round(3).alias("elsewhere")).sort("country", "in_vocab"))
fr = d.filter((pl.col("country") == "France") & (pl.col("state") == "unassigned"))
print("France unassigned: top swapped-in words", fr.group_by("sw_in").len().sort("len", descending=True).head(30).rows())
print("France accepted: top swapped-in words", d.filter((pl.col("country") == "France") & (pl.col("state") == "accepted")).group_by("sw_in").len().sort("len", descending=True).head(30).rows())
print("US unassigned: top swapped-in words", d.filter((pl.col("country") == "US") & (pl.col("state") == "unassigned")).group_by("sw_in").len().sort("len", descending=True).head(20).rows())
if write:
    fr.select("q", "s1", "sw_in", "sw_out", "in_vocab").write_parquet(write)

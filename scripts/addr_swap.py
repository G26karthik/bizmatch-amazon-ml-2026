"""Address-swap check (order-insensitive): accepted pairs whose chosen S1 is NOT at the record's address while another S1 IS
at exactly the record's address (same address token set incl. house number). Copies get one noise operation, so a record
keeping an S1's full address is a name-noised copy of that S1 far more often than a copy of a same-name S1 elsewhere.
Reports counts and name relations by country; with --write stores (q, s1 chosen, s1 at address) for France.

python scripts/addr_swap.py /work/out_fr7 [--write /work/dive/addr_swap.parquet]
"""
import re
import sys
import unicodedata

import polars as pl

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_fmt_str_lengths(55); pl.Config.set_tbl_width_chars(260)
out = sys.argv[1]
write = sys.argv[sys.argv.index("--write") + 1] if "--write" in sys.argv else None
STOP = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "de", "des", "du", "la", "le", "les", "et", "d", "l", "s", "a", "u", "r", "e",
        "inc", "llc", "ltd", "pvt", "private", "limited", "corp", "the", "of", "and", "co", "lp", "llp", "corporation", "incorporated", "company"}


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country", "business_name", "addr_n", "house"]).with_row_index("rid") \
        .with_columns(pl.col("rid").cast(pl.UInt32))
rec = rec.with_columns(pl.Series("akey", [" ".join(sorted(set(re.findall(r"[a-z0-9]+", norm(a))))) if a and h else "" for a, h in zip(rec["addr_n"].to_list(), rec["house"].to_list())]),
                       pl.Series("ntok", [sorted(set(re.findall(r"[a-z0-9]+", norm(n))) - STOP) for n in rec["business_name"].to_list()]))
eid = rec.select("rid", "entity_id")
g = pl.read_csv(f"{out}/matching_results.tsv", separator="\t")
p = g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls("matched_entity_ids") \
     .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
     .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1")
s1a = rec.filter((pl.col("src") == 1) & (pl.col("akey") != "")).group_by("country", "akey").agg(pl.col("rid").alias("here"))
d = p.join(rec.select(pl.col("rid").alias("q"), pl.col("akey").alias("akey_q"), pl.col("ntok").alias("nt_q")), on="q") \
     .join(rec.select(pl.col("rid").alias("s1"), "country", pl.col("akey").alias("akey_s"), pl.col("ntok").alias("nt_s")), on="s1")
d = d.filter((pl.col("akey_q") != "") & (pl.col("akey_q") != pl.col("akey_s")))
d = d.join(s1a.rename({"akey": "akey_q"}), on=["country", "akey_q"], how="inner")
d = d.filter(pl.col("here").list.len() == 1).with_columns(pl.col("here").list.first().alias("s1_here"))
nt = dict(zip(rec["rid"].to_list(), rec["ntok"].to_list()))


def rel(q, a, b):
    tq, ta, tb = set(nt[q]), set(nt[a]), set(nt[b])
    return f"chosen shares {len(tq & ta)}/{len(tq)} words, at-address shares {len(tq & tb)}/{len(tq)}"


d = d.with_columns(pl.Series("sh_chosen", [len(set(nt[q]) & set(nt[a])) / max(len(nt[q]), 1) for q, a in zip(d["q"].to_list(), d["s1"].to_list())]),
                   pl.Series("sh_here", [len(set(nt[q]) & set(nt[b])) / max(len(nt[q]), 1) for q, b in zip(d["q"].to_list(), d["s1_here"].to_list())]))
print("accepted pairs whose chosen S1 is elsewhere while exactly one S1 sits at the record's address, by country:")
print(d.group_by("country").agg(pl.len().alias("n"), (pl.col("sh_here") >= 0.5).mean().round(3).alias("here_shares_half_name"),
                                 (pl.col("sh_chosen") >= 0.99).mean().round(3).alias("chosen_full_name")).sort("country"))
fr = d.filter(pl.col("country") == "France")
names = dict(zip(rec["rid"].to_list(), rec["business_name"].to_list())); addrs = dict(zip(rec["rid"].to_list(), rec["addr_n"].to_list()))
ex = fr.sample(n=min(14, fr.height), seed=4)
for q, a, b in zip(ex["q"].to_list(), ex["s1"].to_list(), ex["s1_here"].to_list()):
    print(f"  REC {names[q]} | {addrs[q]}\n     chosen: {names[a]} | {addrs[a]}\n     at-addr: {names[b]} | {addrs[b]}")
if write:
    fr.select("q", "s1", "s1_here", "sh_chosen", "sh_here").write_parquet(write)

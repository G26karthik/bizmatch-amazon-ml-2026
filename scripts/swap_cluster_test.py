"""Decoy or copy? For S1s with 2+ records at the S1's exact address (order-insensitive tokens) whose name swaps exactly one
word of the S1 name, do those records share the SAME swapped-in word? Copies of one S1 never share noise (generator), so
independent copies would use different words; a decoy business with its own noisy copies shares its name word.
Calibrated on train truth (US/India copies vs orphans), then measured on test France (and US).

python scripts/swap_cluster_test.py
"""
import re
import unicodedata

import polars as pl

LEGAL = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "inc", "llc", "ltd", "pvt", "private", "limited", "corp", "corporation",
         "incorporated", "company", "co", "lp", "llp", "pllc", "pc", "plc", "the", "de", "des", "du", "la", "le", "les", "et", "of", "and", "s", "a", "d", "l"}


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


def run(split, labelled):
    rec = pl.read_parquet(f"/work/base/{split}/records.parquet" if split == "train" else "/work/runfr/test/records.parquet",
                          columns=["entity_id", "src", "country", "business_name", "addr_n", "house"]).with_row_index("rid")
    rec = rec.with_columns(pl.Series("ak", [" ".join(sorted(set(re.findall(r"[a-z0-9]+", norm(a))))) if a and h else "" for a, h in zip(rec["addr_n"].to_list(), rec["house"].to_list())]),
                           pl.Series("nt", [sorted(set(re.findall(r"[a-z0-9]+", norm(n))) - LEGAL) for n in rec["business_name"].to_list()]))
    s1 = rec.filter((pl.col("src") == 1) & (pl.col("ak") != "")).select(pl.col("rid").alias("s1"), "country", "ak", pl.col("nt").alias("nts"), pl.col("entity_id").alias("eid_s1"))
    s1 = s1.join(s1.group_by("country", "ak").len(), on=["country", "ak"]).filter(pl.col("len") == 1).drop("len")  # unique S1 at the address
    q = rec.filter((pl.col("src") != 1) & (pl.col("ak") != "")).select(pl.col("rid").alias("q"), "country", "ak", pl.col("nt").alias("ntq"), "src", pl.col("entity_id").alias("eid_q"))
    d = q.join(s1, on=["country", "ak"])
    sw = []
    for a, b in zip(d["ntq"].to_list(), d["nts"].to_list()):
        A, B = set(a), set(b)
        sw.append(next(iter(A - B)) if len(A - B) == 1 and len(B - A) == 1 and A & B else None)
    d = d.with_columns(pl.Series("sw", sw)).filter(pl.col("sw").is_not_null())
    if labelled:
        gt = pl.read_parquet("/work/data/train_pairs.parquet").select(pl.col("s1").alias("eid_s1"), pl.col("m").alias("eid_q"), pl.lit(True).alias("true"))
        d = d.join(gt, on=["eid_s1", "eid_q"], how="left").with_columns(pl.col("true").fill_null(False))
    g = d.group_by("s1", "country").agg(pl.len().alias("n"), pl.col("sw").n_unique().alias("n_words"), *( [pl.col("true").mean().alias("true_rate")] if labelled else []))
    multi = g.filter(pl.col("n") >= 2)
    for c in sorted(set(multi["country"].to_list())):
        m = multi.filter(pl.col("country") == c)
        shared = m.filter(pl.col("n_words") < pl.col("n"))
        line = f"{split} {c}: one-swap same-address records {d.filter(pl.col('country') == c).height}; S1s with 2+ such records {m.height}; share where a swapped word repeats {shared.height / max(m.height, 1):.3f}"
        if labelled:
            line += f" | true rate in repeat-word groups {shared['true_rate'].mean() if shared.height else 0:.3f} vs distinct-word groups {m.filter(pl.col('n_words') == pl.col('n'))['true_rate'].mean():.3f}"
            line += f" | overall true rate {d.filter(pl.col('country') == c)['true'].mean():.3f}"
        print(line, flush=True)


run("train", True)
run("test", False)

"""Copy or decoy, per swapped-in word, by count conservation. Copies of an S1 are drawn from a fixed distribution
(P(k=1) = 5.4%, P(k=0) = 5.6%, mean 3.46), so an S1 that has a COPY of some type has other copies size-biased
(rarely zero: ~1.6%), while an S1 next to a DECOY has an unbiased copy count (zero ~5.6% + misses). For each France
one-word-swap record on the S1's street (unique S1 on that street with the S1 name key), counts the S1's other accepted
records (all non-swap records) and reports P(other = 0) and mean(other), by (address relation, swapped-in word).
References: US train truth (same address = 96% copies, house + decoy offset = 97% decoys) validates the statistic.

python scripts/swap_verdict.py /work/out_fr7
"""
import re
import sys
import unicodedata

import polars as pl

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(250)
out = sys.argv[1]
LEGAL = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "inc", "llc", "ltd", "pvt", "private", "limited", "corp", "corporation",
         "incorporated", "company", "co", "lp", "llp", "pllc", "pc", "plc", "the", "de", "des", "du", "la", "le", "les", "et", "of", "and", "s", "a", "d", "l"}
OFFS = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}
ONE_PER_STREET = "--strict" in sys.argv


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


def prep(path):
    rec = pl.read_parquet(path, columns=["entity_id", "src", "country", "business_name", "addr_n", "house"]).with_row_index("rid") \
            .with_columns(pl.col("rid").cast(pl.UInt32))
    toks = [set(re.findall(r"[a-z0-9]+", norm(a))) if a else set() for a in rec["addr_n"].to_list()]
    hs = rec["house"].to_list()
    return rec.with_columns(pl.Series("street", [" ".join(sorted(t - {h})) if h else "" for t, h in zip(toks, hs)]),
                            pl.Series("hn", [int(h) if h and h.isdigit() else None for h in hs], dtype=pl.Int64),
                            pl.Series("nt", [sorted(set(re.findall(r"[a-z0-9]+", norm(n))) - LEGAL) for n in rec["business_name"].to_list()]))


def swaps(rec):
    s1 = rec.filter((pl.col("src") == 1) & (pl.col("street") != "") & pl.col("hn").is_not_null())
    if ONE_PER_STREET:
        s1 = s1.join(s1.group_by("country", "street").len(), on=["country", "street"]).filter(pl.col("len") == 1)  # only S1 on its street
    s1 = s1.select(pl.col("rid").alias("s1"), "country", "street", pl.col("hn").alias("hs"), pl.col("nt").alias("nts"), pl.col("entity_id").alias("eid_s1"))
    q = rec.filter((pl.col("src") != 1) & (pl.col("street") != "") & pl.col("hn").is_not_null()) \
           .select(pl.col("rid").alias("q"), "country", "street", pl.col("hn").alias("hq"), pl.col("nt").alias("ntq"), pl.col("entity_id").alias("eid_q"))
    d = q.join(s1, on=["country", "street"]).with_columns((pl.col("hq") - pl.col("hs")).alias("dh"))
    d = d.filter((pl.col("dh") == 0) | pl.col("dh").is_in(list(OFFS)))
    win = []
    for a, b in zip(d["ntq"].to_list(), d["nts"].to_list()):
        A, B = set(a), set(b)
        win.append(next(iter(A - B)) if len(A - B) == 1 and len(B - A) == 1 and A & B else None)
    d = d.with_columns(pl.Series("w_in", win)).filter(pl.col("w_in").is_not_null())
    return d.with_columns(pl.when(pl.col("dh") == 0).then(pl.lit("same")).otherwise(pl.lit("offset")).alias("arel"))


def verdict(d, other, label, words=None):
    """d: swap records (q, s1, arel, w_in); other: per-S1 count of other (non-swap) records."""
    x = d.select("s1", "arel", "w_in").unique().join(other, on="s1", how="left").with_columns(pl.col("n_other").fill_null(0))
    if words is not None:
        x = x.with_columns(pl.when(pl.col("w_in").is_in(words)).then(pl.col("w_in")).otherwise(pl.lit("<other>")).alias("w_in"))
    t = x.group_by("arel", "w_in").agg(pl.len().alias("n_s1"), (pl.col("n_other") == 0).mean().round(4).alias("p_other0"),
                                       pl.col("n_other").mean().round(3).alias("mean_other"))
    print(f"\n=== {label}")
    print(t.filter(pl.col("n_s1") >= 150).sort("arel", "n_s1", descending=[False, True]))


# --- US train truth: other = true copies that are not swap records
tr = prep("/work/base/train/records.parquet")
dtr = swaps(tr.filter(pl.col("country") == "US"))
gt = pl.read_parquet("/work/data/train_pairs.parquet")
e = tr.select("rid", "entity_id")
tp = gt.join(e.rename({"entity_id": "s1", "rid": "s1r"}), on="s1").join(e.rename({"entity_id": "m", "rid": "q"}), on="m").select("q", pl.col("s1r").alias("s1"))
sw_q = dtr.select("q").unique()
oth = tp.join(sw_q, on="q", how="anti").group_by("s1").len().rename({"len": "n_other"})
dtr = dtr.join(tp.with_columns(pl.lit(True).alias("true")), on=["q", "s1"], how="left").with_columns(pl.col("true").fill_null(False))
print("US train swap records: true rate by address relation", dtr.group_by("arel").agg(pl.len(), pl.col("true").mean().round(3)).rows())
verdict(dtr, oth, "US TRAIN TRUTH (other = true non-swap copies)", ["center", "services", "service", "partners"])
verdict(dtr.filter(pl.col("true")), oth, "US TRAIN, true swap records only")
verdict(dtr.filter(~pl.col("true")), oth, "US TRAIN, false swap records only")

# --- test: other = accepted non-swap records
te = prep("/work/runfr/test/records.parquet")
dte = swaps(te)
rec_ids = te.select("rid", "entity_id")
g = pl.read_csv(f"{out}/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls("matched_entity_ids") \
       .join(rec_ids.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
       .join(rec_ids.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1")
oth_te = acc.join(dte.select("q").unique(), on="q", how="anti").group_by("s1").len().rename({"len": "n_other"})
verdict(dte.filter(pl.col("country") == "US"), oth_te, "US TEST (other = accepted non-swap)", ["center", "services", "service", "partners"])
FR_WORDS = ["fils", "cie", "associes", "services", "developpement", "groupe", "france", "compagnie", "frs", "societe", "club", "ecole", "amicale",
            "comite", "sportive", "amis", "centre", "parents", "union", "primaire", "college", "loisirs", "fetes", "sante", "anciens", "pharmacie"]
verdict(dte.filter(pl.col("country") == "France"), oth_te, "FRANCE TEST (other = accepted non-swap)", FR_WORDS)
GEN = ["fils", "cie", "associes", "services", "developpement", "groupe", "france", "frs", "freres", "compagnie"]
fr = dte.filter(pl.col("country") == "France").with_columns(pl.when(pl.col("w_in").is_in(GEN)).then(pl.lit("GENERIC")).otherwise(pl.lit("TYPE/other")).alias("cls"))
verdict(fr.with_columns(pl.col("cls").alias("w_in")), oth_te, "FRANCE TEST grouped: generic vs type words")
fr_acc = fr.join(acc.with_columns(pl.lit(True).alias("acc")), on=["q", "s1"], how="left").with_columns(pl.col("acc").fill_null(False))
for a in (True, False):
    verdict(fr_acc.filter(pl.col("acc") == a).with_columns(pl.col("cls").alias("w_in")), oth_te, f"FRANCE TEST grouped, swap record accepted={a}")
dte.filter(pl.col("country") == "France").select("q", "s1", "arel", "w_in").write_parquet("/work/dive/fr_swaps.parquet")

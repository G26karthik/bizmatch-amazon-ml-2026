"""France: legal-form relation of fr7-accepted pairs (same, drop, add, swap = both have a legal form and they differ),
by name relation (identical key / other) and house relation; cap test per class (rate per 1k S1-source slots with 2-3
other fr7-accepted records vs at the cap; the class's own records excluded from the counts). References: identical key +
legal drop/add at the same house (copies), vocabulary swaps (decoys, accepted in fr7).

python scripts/fr_legal_swap_audit.py
"""
import re
import unicodedata

import polars as pl

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(230)
LEGF = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei"}
LEG = LEGF | {"de", "des", "du", "la", "le", "les", "et", "s", "a", "d", "l"}


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


def legal(n):
    t = norm(n).replace(".", "")
    return frozenset(re.findall(r"[a-z0-9]+", t)) & LEGF


rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "business_name", "src", "country", "addr_n", "house"]).with_row_index("rid") \
        .with_columns(pl.col("rid").cast(pl.UInt32))
nm = rec["business_name"].to_list(); hs = rec["house"].to_list(); ad = rec["addr_n"].to_list(); srcl = rec["src"].to_list()
eid = rec.select("rid", "entity_id")
g = pl.read_csv("/work/out_fr7/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = g.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "") \
       .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
       .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1") \
       .join(rec.filter(pl.col("country") == "France").select(pl.col("rid").alias("s1")), on="s1", how="semi")
cv = pl.read_parquet("/work/lead/fr_contested_vocab.parquet").filter(pl.col("voc1") > 30).select("q", "s1").with_columns(pl.lit(True).alias("vocab"))
acc = acc.join(cv, on=["q", "s1"], how="left").with_columns(pl.col("vocab").fill_null(False))
rows = []
for q, s, v in acc.iter_rows():
    lq, ls = legal(nm[q]), legal(nm[s])
    lrel = "same" if lq == ls else ("drop" if not lq else ("add" if not ls else "swap"))
    A = set(re.findall(r"[a-z0-9]+", norm(nm[q]).replace(".", ""))) - LEG
    B = set(re.findall(r"[a-z0-9]+", norm(nm[s]).replace(".", ""))) - LEG
    nrel = "vocab_swap" if v else ("identical_key" if A == B else "other")
    hrel = "no_addr" if not ad[q] else ("same_house" if hs[q] and hs[q] == hs[s] else ("no_house" if not hs[q] or not hs[s] else "diff_house"))
    rows.append((q, s, lrel, nrel, hrel, srcl[q]))
x = pl.DataFrame(rows, schema=["q", "s1", "lrel", "nrel", "hrel", "src"], orient="row").with_columns(pl.col("src").cast(pl.Int8))
x = x.with_columns(pl.concat_str(["nrel", "lrel", "hrel"], separator="|").alias("cls"))
s1f = rec.filter((pl.col("country") == "France") & (pl.col("src") == 1)).select(pl.col("rid").alias("s1"))
capc = ((pl.col("src") == 2) & (pl.col("n_other") >= 5)) | ((pl.col("src") == 3) & (pl.col("n_other") >= 6))
res = []
for cls in x.group_by("cls").len().filter(pl.col("len") >= 300)["cls"].to_list():
    mine = x.filter(pl.col("cls") == cls)
    oth = x.join(mine.select("q"), on="q", how="anti").group_by("s1", "src").agg(pl.len().alias("n_other"))
    den = pl.concat([s1f.with_columns(pl.lit(s, dtype=pl.Int8).alias("src")) for s in (2, 3)]).join(oth, on=["s1", "src"], how="left") \
            .with_columns(pl.col("n_other").fill_null(0))
    d23, dc = den.filter(pl.col("n_other").is_between(2, 3)).height, den.filter(capc).height
    z = mine.join(oth, on=["s1", "src"], how="left").with_columns(pl.col("n_other").fill_null(0))
    n23, nc = z.filter(pl.col("n_other").is_between(2, 3)).height, z.filter(capc).height
    res.append((cls, mine.height, n23, nc, round(n23 / d23 * 1000, 3), round(nc / dc * 1000, 3), round((nc / dc) / (n23 / d23), 3) if n23 else None))
t = pl.DataFrame(res, schema=["cls", "n", "n23", "ncap", "r23", "rcap", "ratio"], orient="row").sort("n", descending=True)
print(t)
for cls in ("identical_key|swap|same_house", "identical_key|swap|diff_house"):
    print(f"\nexamples {cls}")
    for q, s in x.filter(pl.col("cls") == cls).head(8).select("q", "s1").iter_rows():
        print(f"   {nm[s]!r:44} | {nm[q]!r:44} | {ad[s]} | {ad[q]}")
x.write_parquet("/work/lead/fr_legal_audit.parquet")

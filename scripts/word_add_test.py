"""Pure word-add vs word-swap, per word. Pairs every S2/S3 record containing a watched word with the S1s on the same
street (house removed) whose name key (legal/stop words removed) relates by ADD (record = S1 key + the word) or SWAP
(one S1 word replaced by the word). Train US/India: true rate from labels. Test France: counts, acceptance and cap
test (rate per 1k S1-source slots by accepted others in the same source; copies vanish at the cap 5/6, decoys do not).
Pure decoy words (holding, participations, distribution, international) calibrate the decoy-add rate; groupe and
developpement are both decoy words and generic copy words.

python scripts/word_add_test.py /work/out_fr12
"""
import re
import sys
import unicodedata

import polars as pl

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(230)
out = sys.argv[1]
LEG = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "inc", "llc", "ltd", "pvt", "private", "limited", "corp", "corporation",
       "incorporated", "company", "co", "lp", "llp", "pllc", "pc", "plc", "the", "de", "des", "du", "la", "le", "les", "et", "of", "and", "s", "a", "d", "l"}
OFFS = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}
WATCH = ["holding", "participations", "distribution", "international", "groupe", "developpement", "fils", "cie", "associes", "services",
         "france", "frs", "compagnie", "freres", "southside", "eastgate", "northside", "midtown", "greater", "center", "service", "partners",
         "group", "solutions", "enterprises", "associates"]
WS = set(WATCH)
RX = "(?i)" + "|".join(["holding", "particip", "distribution", "international", "groupe", "d.veloppement", "fils", "cie", "associ", "service",
                        "france", "frs", "compagnie", "fr.res", "southside", "eastgate", "northside", "midtown", "greater", "center",
                        "partners", "group", "solutions", "enterprises"])


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


def prep(df):
    at = [set(re.findall(r"[a-z0-9]+", norm(a))) if a else set() for a in df["addr_n"].to_list()]
    hs = df["house"].to_list()
    street = [" ".join(sorted(t - {h})) if h else " ".join(sorted(t)) for t, h in zip(at, hs)]
    nt = [set(re.findall(r"[a-z0-9]+", norm(n))) - LEG for n in df["business_name"].to_list()]
    hn = [int(h) if h and str(h).isdigit() else None for h in hs]
    return street, nt, hn


def pairs(rec, country):
    s1 = rec.filter((pl.col("src") == 1) & (pl.col("country") == country))
    ot = rec.filter((pl.col("src") != 1) & (pl.col("country") == country) & pl.col("business_name").str.contains(RX))
    s_street, s_nt, s_hn = prep(s1)
    by = {}
    for j, st in enumerate(s_street):
        if st:
            by.setdefault(st, []).append(j)
    o_street, o_nt, o_hn = prep(ot)
    s_rid = s1["rid"].to_list(); o_rid = ot["rid"].to_list(); o_src = ot["src"].to_list(); o_raw = ot["business_name"].to_list()
    res = []
    for i, st in enumerate(o_street):
        A = o_nt[i]
        if not st or not (A & WS):
            continue
        for j in by.get(st, []):
            B = s_nt[j]
            if not B:
                continue
            if B < A and len(A - B) == 1:
                rel, w = "ADD", next(iter(A - B))
            elif len(A - B) == 1 and len(B - A) == 1 and A & B:
                rel, w = "SWAP", next(iter(A - B))
            else:
                continue
            if w not in WS:
                continue
            h1, h2 = o_hn[i], s_hn[j]
            hrel = "nohouse" if h1 is None or h2 is None else ("same" if h1 == h2 else ("offset" if h1 - h2 in OFFS else "other"))
            br = bool(re.search(r"[\[\(]\s*" + w, norm(o_raw[i])))
            res.append((country, o_rid[i], s_rid[j], rel, w, hrel, o_src[i], br))
    return pl.DataFrame(res, schema=["country", "q", "s1", "rel", "word", "hrel", "src", "bracket"], orient="row")


# ---- train truth (full labels, no hiding)
tr = pl.read_parquet("/work/run28/train/records.parquet", columns=["entity_id", "business_name", "country", "src", "addr_n", "house"]) \
       .with_row_index("rid")
gt = pl.read_parquet("/work/data/train_pairs.parquet")
id2 = tr.select("entity_id", "rid")
par = gt.join(id2.rename({"entity_id": "m", "rid": "q"}), on="m").join(id2.rename({"entity_id": "s1", "rid": "p"}), on="s1").select("q", "p")
for c in ("US", "India"):
    x = pairs(tr, c).join(par, on="q", how="left").with_columns((pl.col("p") == pl.col("s1")).fill_null(False).alias("y"))
    t = x.group_by("rel", "word", "hrel").agg(pl.len().alias("n"), pl.col("y").mean().round(3).alias("true_rate"),
                                                pl.col("bracket").mean().round(3).alias("brk")).filter(pl.col("n") >= 30)
    print(f"\n=== train {c}: ADD/SWAP true rates (n>=30)", flush=True)
    print(t.sort("rel", "word", "hrel"), flush=True)
del tr
# ---- test France
te = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "business_name", "country", "src", "addr_n", "house"]) \
       .with_row_index("rid")
x = pairs(te, "France")
eid = te.select("rid", "entity_id")
g = pl.read_csv(f"{out}/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls() \
       .filter(pl.col("matched_entity_ids") != "") \
       .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
       .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1")
srcd = dict(zip(te["rid"].to_list(), te["src"].to_list()))
acc = acc.with_columns(pl.Series("src", [srcd[q] for q in acc["q"].to_list()], dtype=pl.Int8))
x = x.with_columns(pl.col("src").cast(pl.Int8)).join(acc.select("q", "s1").with_columns(pl.lit(True).alias("acc")), on=["q", "s1"], how="left") \
     .with_columns(pl.col("acc").fill_null(False))
xq = x.select("q").unique()
oth = acc.join(xq, on="q", how="anti").group_by("s1", "src").agg(pl.len().alias("n_other"))
x = x.join(oth, on=["s1", "src"], how="left").with_columns(pl.col("n_other").fill_null(0))
s1f = te.filter((pl.col("src") == 1) & (pl.col("country") == "France")).select(pl.col("rid").alias("s1"))
den = pl.concat([s1f.with_columns(pl.lit(s, dtype=pl.Int8).alias("src")) for s in (2, 3)]).join(oth, on=["s1", "src"], how="left") \
        .with_columns(pl.col("n_other").fill_null(0)).group_by("src", "n_other").agg(pl.len().alias("n_slots"))
capc = ((pl.col("src") == 2) & (pl.col("n_other") >= 5)) | ((pl.col("src") == 3) & (pl.col("n_other") >= 6))
x = x.with_columns(capc.alias("at_cap"), pl.col("n_other").is_between(2, 3).alias("n23"))
den_cap = den.filter(capc)["n_slots"].sum()
den_23 = den.filter(pl.col("n_other").is_between(2, 3))["n_slots"].sum()
t = x.group_by("rel", "word", "hrel").agg(pl.len().alias("n"), pl.col("acc").sum().alias("acc"), pl.col("bracket").mean().round(3).alias("brk"),
                                            (pl.col("n23").sum() / den_23 * 1000).round(2).alias("r23"),
                                            (pl.col("at_cap").sum() / den_cap * 1000).round(2).alias("rcap")) \
     .with_columns((pl.col("rcap") / pl.col("r23")).round(2).alias("cap_ratio")).filter(pl.col("n") >= 50)
print("\n=== test France: ADD/SWAP counts, accepted, bracket share, cap test (r23 = per 1k slots with 2-3 others)")
print(t.sort("rel", "word", "hrel"))
x.write_parquet("/work/lead/fr_word_add.parquet")

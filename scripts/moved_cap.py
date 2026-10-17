"""Model-free cap test for same-name records at a moved house on the S1's street (all records, accepted or not).
Population: S2/S3 records whose name key (legal/stop removed) equals an S1's, same street key (address words without
numbers), both houses parsed, house different: class = negative / positive non-offset / decoy offset. Rate per 1k
S1-source slots by n_same = number of fr14-accepted records of the S1 in that source at the S1's exact house (records of
the class excluded). Copies vanish as n_same reaches the cap (5 S2 / 6 S3); decoys stay flat. US test = reference.
Also the acceptance share of each class by n_same.

uv run --no-project --with polars==1.44.2 python scripts/moved_cap.py
"""
import re
import unicodedata

import polars as pl

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(230)
LEG = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "inc", "llc", "ltd", "pvt", "private", "limited", "corp", "corporation",
       "incorporated", "company", "co", "lp", "llp", "pllc", "pc", "plc", "the", "of", "and", "de", "des", "du", "la", "le", "les", "et", "s", "a", "d", "l"}
OFFS = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}


def asc(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


def nkey(s):
    return " ".join(sorted(set(re.findall(r"[a-z0-9]+", asc(s).replace(".", ""))) - LEG))


def addr(s):
    a = asc(s)
    m = re.search(r"\b(\d{1,5})\b", a)
    h = int(m.group(1)) if m else None
    st = " ".join(sorted(set(re.findall(r"[a-z]{2,}", a))))
    return h, st


t = pl.concat([pl.read_parquet(f"local_data/test_source{s}.parquet").with_columns(pl.lit(s).cast(pl.Int8).alias("src")) for s in (1, 2, 3)])
g = pl.read_csv("output/v28q_fr14/matching_results.tsv", separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
acc = g.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
accm = dict(zip(acc["matched_entity_ids"].to_list(), acc["source1_entity_id"].to_list()))
for country in ("France", "US"):
    x = t.filter(pl.col("country") == country)
    ids = x["entity_id"].to_list(); nm = x["business_name"].to_list(); ad = x["business_address"].to_list(); src = x["src"].to_list()
    info = [addr(a) if a and a.strip() else (None, "") for a in ad]
    keys = [nkey(n) for n in nm]
    s1idx = {}
    for i in range(len(ids)):
        if src[i] == 1 and info[i][0] is not None and info[i][1]:
            s1idx.setdefault((keys[i], info[i][1]), []).append(i)
    s1house = {ids[i]: info[i][0] for i in range(len(ids)) if src[i] == 1}
    rows = []
    for i in range(len(ids)):
        if src[i] == 1 or info[i][0] is None or not info[i][1]:
            continue
        for j in s1idx.get((keys[i], info[i][1]), []):
            d = info[i][0] - info[j][0]
            cls = "same" if d == 0 else ("offset" if d in OFFS else ("negative" if d < 0 else "pos_other"))
            rows.append((ids[i], ids[j], cls, src[i], accm.get(ids[i]) == ids[j], accm.get(ids[i]) is not None))
    y = pl.DataFrame(rows, schema=["m", "s1", "cls", "src", "acc_here", "assigned"], orient="row")
    # n_same: accepted records of the S1 in the same source at its exact house (from the 'same' class, accepted)
    same_acc = y.filter((pl.col("cls") == "same") & pl.col("acc_here")).group_by("s1", "src").agg(pl.len().alias("n_same"))
    y = y.join(same_acc, on=["s1", "src"], how="left").with_columns(pl.col("n_same").fill_null(0))
    s1s = x.filter(pl.col("src") == 1).select(pl.col("entity_id").alias("s1"))
    den = pl.concat([s1s.with_columns(pl.lit(s, dtype=pl.Int8).alias("src")) for s in (2, 3)]).join(same_acc, on=["s1", "src"], how="left") \
            .with_columns(pl.col("n_same").fill_null(0)).group_by("src", "n_same").agg(pl.len().alias("slots"))
    tt = y.filter(pl.col("cls") != "same").group_by("cls", "src", "n_same").agg(pl.len().alias("n"), pl.col("acc_here").mean().round(3).alias("acc")) \
          .join(den, on=["src", "n_same"]).with_columns((pl.col("n") / pl.col("slots") * 1000).round(3).alias("per1k"))
    print(f"\n=== {country}: same-name moved-house records per 1k S1-source slots by n_same (accepted exact-house copies of the S1 in that source)")
    print(tt.pivot(on="n_same", index=["cls", "src"], values="per1k", sort_columns=True).sort("cls", "src"))
    print("acceptance share by n_same:")
    print(tt.pivot(on="n_same", index=["cls", "src"], values="acc", sort_columns=True).sort("cls", "src"))
    print("class totals:", y.group_by("cls").agg(pl.len(), pl.col("acc_here").sum().alias("accepted")).sort("cls").rows())
    if country == "France":
        y.write_parquet("local_data/work/fr_moved.parquet")

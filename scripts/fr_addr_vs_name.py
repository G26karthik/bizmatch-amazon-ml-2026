"""France vs US: for accepted pairs, does the record's address match the chosen S1's address? And for records whose chosen
S1 is at a DIFFERENT address, is there another S1 at the record's own exact address (address match beats name match in
France, where names are generic)?

python scripts/fr_addr_vs_name.py /work/out_fr7 [--write /work/dive/fr_readdr.parquet]
"""
import re
import sys
import unicodedata

import polars as pl

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_fmt_str_lengths(60); pl.Config.set_tbl_width_chars(260)
out = sys.argv[1]
write = sys.argv[sys.argv.index("--write") + 1] if "--write" in sys.argv else None
rec = pl.read_parquet("/work/runfr/test/records.parquet", columns=["entity_id", "src", "country", "business_name", "business_address", "addr_n", "house"]) \
        .with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))


def street(a):
    a = unicodedata.normalize("NFKD", a or "").encode("ascii", "ignore").decode().lower()
    return " ".join(re.findall(r"[a-z]+", a))


rec = rec.with_columns(pl.Series("st", [street(a) for a in rec["addr_n"].to_list()]))
eid = rec.select("rid", "entity_id")
g = pl.read_csv(f"{out}/matching_results.tsv", separator="\t")
p = g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls("matched_entity_ids") \
     .join(eid.rename({"entity_id": "source1_entity_id", "rid": "s1"}), on="source1_entity_id") \
     .join(eid.rename({"entity_id": "matched_entity_ids", "rid": "q"}), on="matched_entity_ids").select("q", "s1")
cols = ["rid", "country", "business_name", "addr_n", "house", "st"]
d = p.join(rec.select(cols).rename({c: c + "_q" for c in cols if c != "rid"}).rename({"rid": "q"}), on="q") \
     .join(rec.select(cols).rename({c: c + "_s" for c in cols if c != "rid"}).rename({"rid": "s1"}), on="s1")


def lvl(aq, hq, stq, as_, hs, sts):
    if not aq or not as_:
        return "missing"
    if aq == as_:
        return "exact"
    if hq and hs and hq == hs and stq == sts:
        return "house+street"
    if stq == sts:
        return "street only"
    if hq and hs and hq == hs:
        return "house only"
    return "different"


d = d.with_columns(pl.Series("addr_rel", [lvl(*r) for r in zip(*[d[c].to_list() for c in ("addr_n_q", "house_q", "st_q", "addr_n_s", "house_s", "st_s")])]))
print("accepted pairs: address relation between record and chosen S1, by country (share):")
print(d.group_by("country_s", "addr_rel").len().with_columns((pl.col("len") / pl.col("len").sum().over("country_s")).round(4).alias("share"))
       .sort("country_s", "len", descending=[False, True]))
# France: records at a different address than their chosen S1 -> is there an S1 at the record's own exact address?
s1a = rec.filter((pl.col("src") == 1) & (pl.col("addr_n") != "") & pl.col("house").is_not_null() & (pl.col("house") != "")) \
         .group_by("country", "addr_n").agg(pl.col("rid").alias("s1_here"), pl.col("business_name").alias("names_here"))
for c in ("France", "US"):
    x = d.filter((pl.col("country_s") == c) & pl.col("addr_rel").is_in(["different", "street only", "house only"]))
    y = x.join(s1a.rename({"country": "country_s", "addr_n": "addr_n_q"}), on=["country_s", "addr_n_q"], how="inner") \
         .filter(~pl.col("s1_here").list.contains(pl.col("s1")))
    print(f"{c}: accepted pairs whose chosen S1 is at another address {x.height}; of these, an S1 exists at the record's exact address: {y.height}")
    if c == "France":
        print(y.sample(n=min(15, y.height), seed=2).select("business_name_q", "addr_n_q", "business_name_s", "addr_n_s", "names_here"))
        if write:
            y.select("q", "s1", "s1_here").write_parquet(write)

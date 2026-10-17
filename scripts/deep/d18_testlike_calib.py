"""Deep dive 18: calibration of production probabilities on validation uncertain pairs when the records that only
exist because of the 19% S1 hiding (copies of hidden S1s, 'H records') are removed (test has none: empty-address
records per S1 are equal in train and test). By record type (empty address or not) and name-key class of the record
(n_vis: visible S1s sharing the record's nkey).

uv run --no-project --with polars==1.44.2 python scripts/deep/d18_testlike_calib.py
"""
import re
import unicodedata

import polars as pl

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(220)
SEED, HIDE, VAL_MOD = 42, 19, 10
LEG = {"inc", "llc", "ltd", "limited", "private", "pvt", "corp", "corporation", "co", "company", "incorporated", "lp", "llp",
       "pc", "plc", "the", "and", "of", "sa", "sas", "sarl", "sasu", "eurl", "sci", "snc", "l", "c", "p", "d", "s", "a", "de", "du", "des", "la", "le", "les", "et"}


def nkey(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    return " ".join(sorted(set(re.findall(r"\w+", "".join(ch for ch in s if not unicodedata.combining(ch)))) - LEG))


rec = pl.concat([pl.read_parquet(f"local_data/train_source{s}.parquet").with_columns(pl.lit(s).cast(pl.Int8).alias("src")) for s in (1, 2, 3)]).with_row_index("rid")
rec = rec.with_columns(pl.col("rid").cast(pl.UInt32), ((pl.col("src") == 1) & ((pl.col("entity_id").hash(SEED + 77) % 100) < HIDE)).alias("hidden"),
                       (pl.col("business_address").fill_null("").str.strip_chars() == "").alias("empty"),
                       pl.col("business_name").map_elements(nkey, return_dtype=pl.String).alias("k"))
tp = pl.read_parquet("local_data/train_pairs.parquet")
hid = rec.filter(pl.col("hidden")).select(pl.col("entity_id").alias("s1"))
Hm = set(tp.join(hid, on="s1")["m"].to_list())
id2rid = rec.select("entity_id", "rid")
par = tp.join(id2rid.rename({"entity_id": "m", "rid": "q"}), on="m").join(id2rid.rename({"entity_id": "s1", "rid": "p"}), on="s1").select("q", "p")
vis = rec.filter((pl.col("src") == 1) & ~pl.col("hidden")).group_by("country", "k").len().rename({"len": "n_vis"})
info = rec.select(pl.col("rid").alias("q"), "entity_id", "empty", "country", "k").join(vis, on=["country", "k"], how="left").with_columns(pl.col("n_vis").fill_null(0).clip(0, 2))
u = pl.read_parquet("local_data/llm/unc_val.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
u = u.join(info, on="q").join(par, on="q", how="left").with_columns((pl.col("p") == pl.col("s1")).fill_null(False).alias("y"),
                                                                  pl.col("entity_id").is_in(list(Hm)).alias("H"))
print("uncertain val pairs", u.height, "H records among them", u["H"].sum())
u = u.with_columns(pl.col("prob").cut([0.1, 0.3, 0.5, 0.7, 0.9]).alias("pbin"))
for tl in (False, True):
    d = u.filter(~pl.col("H")) if tl else u
    print("\n", "TEST-LIKE (H removed)" if tl else "STANDARD (with H)")
    print(d.group_by("empty", "n_vis", "pbin").agg(pl.len(), pl.col("prob").mean().round(3).alias("mean_p"), pl.col("y").mean().round(3).alias("P_true"))
          .sort("empty", "n_vis", "pbin"))

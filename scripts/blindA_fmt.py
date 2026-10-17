"""blindA: formatting fingerprints of raw strings, matched vs unmatched (train), and train vs test rates."""
import polars as pl, numpy as np, unicodedata
O = "/work/agent_blindA"
pairs = pl.read_parquet(f"{O}/pairs_train.parquet")
matched = pairs.select(pl.col("m").alias("entity_id")).unique().with_columns(pl.lit(True).alias("matched"))

def feats(df):
    n = pl.col("name").fill_null(""); a = pl.col("addr").fill_null("")
    ex = []
    for c, col in [("n", n), ("a", a)]:
        ex += [
            col.str.contains(r"^\s").alias(f"{c}_lead_ws"),
            col.str.contains(r"\s$").alias(f"{c}_trail_ws"),
            col.str.contains(r"  ").alias(f"{c}_dblspace"),
            col.str.contains(" ").alias(f"{c}_nbsp"),
            col.str.contains(r"[ -‏  　﻿]").alias(f"{c}_uspace_zw"),
            col.str.contains(r"[̀-ͯ]").alias(f"{c}_combining"),
            col.str.contains(r"[^\x00-\x7f]").alias(f"{c}_nonascii"),
            col.str.contains(r"[ऀ-෿]").alias(f"{c}_indic"),
            col.str.contains(r"[À-ɏ]").alias(f"{c}_latin_ext"),
            (col == col.str.to_uppercase()).and_(col.str.contains(r"[A-Za-z]")).alias(f"{c}_allupper"),
            (col == col.str.to_lowercase()).and_(col.str.contains(r"[A-Za-z]")).alias(f"{c}_alllower"),
            col.str.contains(r"^[^\w\s]").alias(f"{c}_lead_punct"),
            col.str.contains(r"[^\w\s]$").alias(f"{c}_trail_punct"),
            col.str.contains(r"[\(\)\[\]\{\}]").alias(f"{c}_bracket"),
            col.str.contains(r"[\"`‘’“”]").alias(f"{c}_quotes"),
            col.str.contains(r"[,.;:\-]{2,}").alias(f"{c}_rep_punct"),
            col.str.contains(r"[^\w\s,.'&\-/#()+]").alias(f"{c}_odd_char"),
            col.str.contains(r"\s[,.]").alias(f"{c}_space_before_punct"),
            col.str.contains(r",\S").alias(f"{c}_comma_nospace"),
            (col == "").alias(f"{c}_empty"),
            col.str.contains(r"\d").alias(f"{c}_digit"),
            col.str.contains(r"(?i)\.com|www\.|\.in\b|\.fr\b|\.net|\.org").alias(f"{c}_web"),
        ]
    ex += [a.str.count_matches(",").alias("a_ncomma"), n.str.len_chars().alias("n_len"), a.str.len_chars().alias("a_len"),
           n.str.count_matches(" ").alias("n_nsp")]
    df = df.with_columns(ex)
    nfc_n = df["name"].fill_null("").map_elements(lambda x: unicodedata.is_normalized("NFC", x), return_dtype=pl.Boolean)
    nfc_a = df["addr"].fill_null("").map_elements(lambda x: unicodedata.is_normalized("NFC", x), return_dtype=pl.Boolean)
    nfkc_n = df["name"].fill_null("").map_elements(lambda x: unicodedata.is_normalized("NFKC", x), return_dtype=pl.Boolean)
    df = df.with_columns((~nfc_n).alias("n_not_nfc"), (~nfc_a).alias("a_not_nfc"), (~nfkc_n).alias("n_not_nfkc"))
    return df

bool_cols = None
res = []
for split in ["train", "test"]:
    for k in [1, 2, 3]:
        df = pl.read_parquet(f"{O}/raw_{split}_s{k}.parquet")
        df = feats(df)
        if split == "train" and k > 1:
            df = df.join(matched, on="entity_id", how="left").with_columns(pl.col("matched").fill_null(False))
        else:
            df = df.with_columns(pl.lit(None, dtype=pl.Boolean).alias("matched"))
        if bool_cols is None:
            bool_cols = [c for c, t in df.schema.items() if t == pl.Boolean and c != "matched"]
        df.drop(["name", "addr"]).write_parquet(f"{O}/fmt_{split}_s{k}.parquet")
        for ctry in df["country"].unique().sort().to_list():
            d = df.filter(pl.col("country") == ctry)
            if split == "train" and k > 1:
                for mm in [True, False]:
                    dd = d.filter(pl.col("matched") == mm)
                    r = {"split": split, "src": k, "country": ctry, "grp": "M" if mm else "U", "n": dd.height}
                    r.update({c: dd[c].mean() for c in bool_cols})
                    r.update({"a_ncomma": dd["a_ncomma"].mean(), "n_len": dd["n_len"].mean(), "a_len": dd["a_len"].mean()})
                    res.append(r)
            else:
                r = {"split": split, "src": k, "country": ctry, "grp": "all", "n": d.height}
                r.update({c: d[c].mean() for c in bool_cols})
                r.update({"a_ncomma": d["a_ncomma"].mean(), "n_len": d["n_len"].mean(), "a_len": d["a_len"].mean()})
                res.append(r)
        print("done", split, k, flush=True)
R = pl.DataFrame(res)
R.write_parquet(f"{O}/fmt_rates.parquet")
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(250)
T = R.unpivot(index=["split", "src", "country", "grp", "n"], variable_name="feat", value_name="rate")
T = T.with_columns((pl.col("split") + "_S" + pl.col("src").cast(pl.Utf8) + "_" + pl.col("grp")).alias("key"))
for ctry in ["US", "India", "France"]:
    W = T.filter(pl.col("country") == ctry).pivot(on="key", index="feat", values="rate")
    print("\n#####", ctry)
    print(W.with_columns(pl.all().exclude("feat").round(4)))

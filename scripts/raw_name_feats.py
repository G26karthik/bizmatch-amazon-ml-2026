"""Raw-spelling features for zone pairs (added to the stacker's raw features): similarity of the raw business names
(case-sensitive and case-insensitive) and whether the legal-suffix form agrees ("Pvt. Ltd." vs "Private Limited").
Name normalisation drops legal words, but the generator keeps an S1's suffix style in its copies, which helps pick the
right S1 among namesakes.

python scripts/raw_name_feats.py --base /work/base --raw /work/rawf --out /work/rawf2
"""
import argparse
import re
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
from unidecode import unidecode

SUF = re.compile(r"(private limited|pvt\.? ?ltd\.?|pvt|limited|ltd\.?|l\.?l\.?c\.?|inc\.?|incorporated|corp\.?|corporation"
                 r"|co\.|company|llp|p\.?c\.?|pllc|s\.?a\.?r\.?l\.?|s\.?a\.?s\.?u?|e\.?u\.?r\.?l\.?|s\.?c\.?i\.?)\s*[\]\)]?\s*$", re.I)


def suffix(s: str) -> str:
    m = SUF.search(unidecode(s or "").strip())
    return re.sub(r"[^a-z]", "", m.group(1).lower()) if m else ""


ap = argparse.ArgumentParser()
ap.add_argument("--base", type=Path, required=True)
ap.add_argument("--raw", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
for split, name in (("train", "val"), ("test", "test")):
    rec = pl.read_parquet(a.base / split / "records.parquet", columns=["business_name"]).with_row_index("rid") \
            .with_columns(pl.col("rid").cast(pl.UInt32))
    rec = rec.with_columns(pl.Series("suf", [suffix(x) for x in rec["business_name"].to_list()]))
    r = pl.read_parquet(a.raw / f"raw_{name}.parquet")
    d = r.select("q", "s1").join(rec.rename({"rid": "q", "business_name": "qn", "suf": "qs"}), on="q", how="left") \
         .join(rec.rename({"rid": "s1", "business_name": "sn", "suf": "ss"}), on="s1", how="left")
    qn, sn = d["qn"].fill_null("").to_list(), d["sn"].fill_null("").to_list()
    d = d.with_columns(
        pl.Series("raw_name_ratio", cpdist(qn, sn, scorer=fuzz.ratio, workers=-1).astype(np.float32)),
        pl.Series("raw_name_ratio_ci", cpdist([x.lower() for x in qn], [x.lower() for x in sn], scorer=fuzz.ratio, workers=-1).astype(np.float32)),
        ((pl.col("qs") == pl.col("ss")) & (pl.col("qs") != "")).cast(pl.Int8).alias("suffix_eq"),
        ((pl.col("qs") == "") & (pl.col("ss") != "")).cast(pl.Int8).alias("suffix_dropped"))
    r.join(d.select("q", "s1", "raw_name_ratio", "raw_name_ratio_ci", "suffix_eq", "suffix_dropped"), on=["q", "s1"], how="left") \
     .write_parquet(a.out / f"raw_{name}.parquet")
    print(name, r.height, "pairs; suffix_eq share", round(float(d["suffix_eq"].mean()), 3), flush=True)

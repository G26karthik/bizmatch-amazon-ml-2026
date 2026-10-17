"""Legal-type change features for zone pairs (added to the stacker's raw features).

Generator rules (training labels): swapping the legal type (LLC <-> Inc <-> Corp <-> Ltd, Pvt Ltd <-> LLP, SARL <-> SAS
...) is 86-99% a decoy, while dropping it or writing it differently (Inc -> Incorporated, Private Limited -> Pvt Ltd)
is ordinary copy noise. Name normalisation removes legal words, so the tree stack cannot see the difference.
Columns: legal_swap, legal_drop, legal_add, legal_restyle (same type, other spelling), legal_same.

python scripts/legal_feats.py --base /work/base --raw /work/rawf --out /work/rawf3
"""
import argparse
import re
from pathlib import Path

import polars as pl
from unidecode import unidecode

# canonical legal type for the trailing legal phrase of a name (checked longest-first)
TYPES = [
    (r"private[\s\-]*limited|private[\s\-]*ltd\.?|pvt\.?[\s\.\-]*ltd\.?|pvt\.?[\s\-]*limited|\(p\)[\s\-]*ltd\.?|p\.?[\s\-]*ltd\.?", "PVTLTD"),
    (r"l\.?l\.?p\.?", "LLP"), (r"p\.?l\.?l\.?c\.?", "PLLC"), (r"l\.?l\.?c\.?", "LLC"),
    (r"incorporated|inc\.?", "INC"), (r"corporation|corp\.?", "CORP"), (r"limited|ltd\.?", "LTD"),
    (r"company|co\.", "CO"), (r"l\.?p\.?", "LP"), (r"p\.?c\.?", "PC"), (r"o\.?p\.?c\.?", "OPC"),
    (r"private|pvt\.?", "PRIVATE"), (r"group", "GROUP"),
    (r"s\.?a\.?s\.?u\.?", "SASU"), (r"s\.?a\.?r\.?l\.?", "SARL"), (r"s\.?a\.?s\.?", "SAS"), (r"e\.?u\.?r\.?l\.?", "EURL"),
    (r"s\.?c\.?i\.?", "SCI"), (r"s\.?n\.?c\.?", "SNC"), (r"e\.?i\.?r\.?l\.?", "EIRL"), (r"e\.?i\.?", "EI"), (r"s\.?a\.?", "SA"),
]
SUFFIX = re.compile(r"(?:^|[\s,\[\(\-&])(" + "|".join(p for p, _ in TYPES) + r")\s*[\]\)\.]*\s*$", re.I)


def legal(name: str) -> tuple[str, str]:
    """(canonical type, surface form) of the trailing legal phrase, or ('', '')."""
    s = unidecode(name or "").strip()
    m = SUFFIX.search(s)
    if not m:
        return "", ""
    surf = m.group(1).lower()
    for p, t in TYPES:
        if re.fullmatch(p, surf, re.I):
            return t, re.sub(r"[^a-z()]", "", surf)
    return "", ""


ap = argparse.ArgumentParser()
ap.add_argument("--base", type=Path, required=True)
ap.add_argument("--raw", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
for split, name in (("train", "val"), ("test", "test")):
    rec = pl.read_parquet(a.base / split / "records.parquet", columns=["business_name"]).with_row_index("rid") \
            .with_columns(pl.col("rid").cast(pl.UInt32))
    lt = [legal(x) for x in rec["business_name"].to_list()]
    rec = rec.with_columns(pl.Series("lt", [t for t, _ in lt]), pl.Series("ls", [s for _, s in lt])).drop("business_name")
    r = pl.read_parquet(a.raw / f"raw_{name}.parquet")
    d = r.select("q", "s1").join(rec.rename({"rid": "q", "lt": "qt", "ls": "qs"}), on="q", how="left") \
         .join(rec.rename({"rid": "s1", "lt": "st", "ls": "ss"}), on="s1", how="left")
    d = d.with_columns(((pl.col("qt") != "") & (pl.col("st") != "") & (pl.col("qt") != pl.col("st"))).cast(pl.Int8).alias("legal_swap"),
                       ((pl.col("qt") == "") & (pl.col("st") != "")).cast(pl.Int8).alias("legal_drop"),
                       ((pl.col("qt") != "") & (pl.col("st") == "")).cast(pl.Int8).alias("legal_add"),
                       ((pl.col("qt") != "") & (pl.col("qt") == pl.col("st")) & (pl.col("qs") != pl.col("ss"))).cast(pl.Int8).alias("legal_restyle"),
                       ((pl.col("qt") != "") & (pl.col("qt") == pl.col("st")) & (pl.col("qs") == pl.col("ss"))).cast(pl.Int8).alias("legal_same"))
    ids = {t: i + 1 for i, (_, t) in enumerate(TYPES)}
    d = d.with_columns(pl.col("qt").replace_strict(ids, default=0).cast(pl.Int16).alias("legal_q_type"),
                       pl.col("st").replace_strict(ids, default=0).cast(pl.Int16).alias("legal_s_type"))
    out = r.join(d.select("q", "s1", "legal_swap", "legal_drop", "legal_add", "legal_restyle", "legal_same", "legal_q_type",
                          "legal_s_type"), on=["q", "s1"], how="left")
    out.write_parquet(a.out / f"raw_{name}.parquet")
    print(name, r.height, "pairs;", {c: round(float(d[c].mean()), 4) for c in ("legal_swap", "legal_drop", "legal_add", "legal_restyle", "legal_same")}, flush=True)

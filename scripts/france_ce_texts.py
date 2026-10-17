"""Cross-encoder input texts for France test pairs with France-specific noise removed (original casing kept, so the
text looks like the cross-encoders' training data):
* name: French noise words that replace an S1 word in true copies are dropped: developpement, groupe, associes and the
  "& Fils" / "Et Fils" / "& Associes" connectors. Decoy words (participations, holding, distribution, international)
  are kept: they mark decoys, which the models already reject.
* address: region / department components dropped (S1 uses regions, S2/S3 often departments), "N°"/"No" before a
  house number removed, "St-"/"St.-" expanded to "Saint-".
Both sides of every pair are cleaned the same way. Output: q, s1, text_a (S1), text_b (record) for every France pair
scored by any cross-encoder, plus the original texts for reference.

python scripts/france_ce_texts.py --base /work/base --ce /work/ce_wide /work/ce_wide_base /work/ce_wide_base2 /work/ce_large --out /work/fr_ce_texts.parquet
"""
import argparse
import re
from pathlib import Path

import polars as pl
from unidecode import unidecode

NAME_DROP = {"developpement", "groupe", "associes", "fils"}
REGIONS = {"hauts de france", "nouvelle aquitaine", "pays de la loire", "nord", "gironde", "loire atlantique",
           "pas de calais"}
_WORD = re.compile(r"[^\s,()\[\]]+")
_CONNECT_END = re.compile(r"(?:\s*(?:&|\bet\b|\band\b))+\s*$", re.I)
_CONNECT_DUP = re.compile(r"(?:\s*&\s*){2,}")
_NUM = re.compile(r"\bN(?:°|o|º)\.?\s*(?=\d)", re.I)
_SAINT = re.compile(r"\b(S)(?:t|T)\.?-", re.I)
_SAINTE = re.compile(r"\b(S)(?:te|TE)\.?-", re.I)


def _key(tok: str) -> str:
    return re.sub(r"[^a-z]", "", unidecode(tok).lower())


def clean_name(s: str) -> str:
    s = s or ""
    out = _WORD.sub(lambda m: "" if _key(m.group(0)) in NAME_DROP else m.group(0), s)
    out = re.sub(r"\s+(?:&|et|ET|Et)\s*(?=$|[\s,)\]])", " ", out)  # connector left dangling by a dropped word
    out = _CONNECT_DUP.sub(" & ", out)
    out = _CONNECT_END.sub("", out)
    out = re.sub(r"\(\s*\)|\[\s*\]", "", out)
    out = " ".join(out.split())
    return out if out else s


def clean_addr(s: str) -> str:
    parts = [p.strip() for p in (s or "").split(",")]
    keep = [p for p in parts if p and " ".join(re.sub(r"[^a-z]+", " ", unidecode(p).lower()).split()) not in REGIONS]
    out = ", ".join(keep)
    out = _NUM.sub("", out)
    out = _SAINTE.sub(lambda m: ("S" if m.group(1) == "S" else "s") + "ainte-", out)
    out = _SAINT.sub(lambda m: ("S" if m.group(1) == "S" else "s") + "aint-", out)
    return " ".join(out.split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--ce", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rec = pl.read_parquet(a.base / "test" / "records.parquet", columns=["business_name", "business_address", "country"]) \
            .with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
    fr = rec.filter(pl.col("country") == "France")
    cn = [clean_name(x) for x in fr["business_name"].to_list()]
    ca = [clean_addr(x) for x in fr["business_address"].to_list()]
    fr = fr.with_columns(pl.Series("t_new", [f"{n} | {ad}" for n, ad in zip(cn, ca)]),
                         (pl.col("business_name") + " | " + pl.col("business_address")).alias("t_old"))
    print(f"France records {fr.height}; text changed {(fr['t_new'] != fr['t_old']).sum()}")
    for r in fr.filter(pl.col("t_new") != pl.col("t_old")).sample(12, seed=1).iter_rows(named=True):
        print(f"  {r['t_old']}\n  -> {r['t_new']}")
    pairs = pl.concat([pl.read_parquet(c / "ce_scores_test.parquet", columns=["q", "s1"]) for c in a.ce]).unique() \
              .with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
    t = fr.select("rid", "t_new", "t_old")
    p = pairs.join(t.rename({"rid": "s1", "t_new": "text_a", "t_old": "old_a"}), on="s1") \
             .join(t.rename({"rid": "q", "t_new": "text_b", "t_old": "old_b"}), on="q")
    p = p.filter((pl.col("text_a") != pl.col("old_a")) | (pl.col("text_b") != pl.col("old_b")))  # only pairs whose text changed
    p.select("q", "s1", "text_a", "text_b").sort("q", "s1").write_parquet(a.out)
    # original texts of the same pairs: scored in the same environment, so only the cleaning effect is transferred
    p.select("q", "s1", pl.col("old_a").alias("text_a"), pl.col("old_b").alias("text_b")).sort("q", "s1")         .write_parquet(a.out.with_name(a.out.stem + "_orig.parquet"))
    print(f"France pairs to rescore (text changed): {p.height}")


if __name__ == "__main__":
    main()

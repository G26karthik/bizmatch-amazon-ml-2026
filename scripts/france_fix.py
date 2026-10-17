"""Re-score France test pairs with France-cleaned text (ber.france), keeping the trained models and candidate set.

France rows of the test records are re-normalised; string features of the existing France candidate pairs are
recomputed; stage-1/2/3 probabilities (p3) are recomputed with the saved models. US / India rows are untouched.
Writes <out>/test/records.parquet, <out>/test/pruned/France_*.parquet, <out>/scores_test.parquet (all countries, France
rows replaced), <out>/raw_test.parquet (zone raw features with France rows replaced).

python scripts/france_fix.py --base /work/base --pruned /work/frfix/pruned --emb /work/frfix/emb_cos_pruned_test.parquet \
    --models /work/frfix/run20_models --scores /work/run20/scores_test.parquet --raw /work/rawf/raw_test.parquet --out /work/runfr
"""
import argparse
import math
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import polars as pl

sys.path.insert(0, "src")
from ber.france import fr_addr, fr_name  # noqa: E402
from ber.pipeline import NPROC, STR_COLS, _norm_chunk, _pair_strings, _records_for_features, log  # noqa: E402

RAW_COLS = ["cos_a", "cos_n", "n_cand", "rel_top1", "gap12", "name_ratio", "name_tsr", "name_pr", "name_jw", "skel_ratio",
            "skel_tsr", "addr_ratio", "addr_tsr", "addr_pr", "street_tsr", "street_ratio", "house_sim", "house_eq",
            "house_miss", "house_absdiff", "postal_eq", "postal_miss", "num_jac", "num_inter", "first_tok_eq", "q_nonascii",
            "q_addr_empty", "q_src3", "len_name1", "len_name2", "len_addr1", "len_addr2", "pA"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--pruned", type=Path, required=True)
    ap.add_argument("--emb", type=Path, required=True)
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    (a.out / "test" / "pruned").mkdir(parents=True, exist_ok=True)

    # 1. France records re-normalised with the France cleaning in front of the usual normaliser
    rec = pl.read_parquet(a.base / "test" / "records.parquet")
    fr = rec.with_row_index("rid").filter(pl.col("country") == "France")
    rows = [(fr_name(n), fr_addr(ad)) for n, ad in zip(fr["business_name"].to_list(), fr["business_address"].to_list())]
    step = math.ceil(len(rows) / (NPROC * 8))
    with Pool(NPROC) as p:
        res = [r for part in p.map(_norm_chunk, [rows[i:i + step] for i in range(0, len(rows), step)]) for r in part]
    cols = list(zip(*res))
    new = fr.select("rid").with_columns(
        pl.Series("name_n", cols[0]), pl.Series("addr_n", cols[1]), pl.Series("house", cols[2]),
        pl.Series("postal", cols[3]), pl.Series("nums", cols[4], dtype=pl.List(pl.String)),
        pl.Series("name_sk", cols[6]), pl.Series("addr_w", cols[7]))
    upd = ["name_n", "addr_n", "house", "postal", "nums", "name_sk", "addr_w"]
    rec = rec.with_row_index("rid").join(new.rename({c: f"_{c}" for c in upd}), on="rid", how="left").with_columns(
        *[pl.coalesce(f"_{c}", c).alias(c) for c in upd]).drop([f"_{c}" for c in upd] + ["rid"])
    rec.write_parquet(a.out / "test" / "records.parquet")
    chg = fr.select("name_n", "addr_n").with_columns(new["name_n"].alias("nn"), new["addr_n"].alias("na"))
    log(f"France records re-normalised: {fr.height}; names changed {(chg['name_n'] != chg['nn']).sum()}, "
        f"addresses changed {(chg['addr_n'] != chg['na']).sum()}")
    print(chg.filter((pl.col("name_n") != pl.col("nn")) | (pl.col("addr_n") != pl.col("na"))).head(8))

    # 2. string features of the existing France candidate pairs, recomputed
    rf = _records_for_features(a.out, "test")
    parts = sorted(a.pruned.glob("France_*.parquet"))
    for f in parts:
        d = pl.read_parquet(f)
        s = _pair_strings(d.select("q", "s1"), rf)
        d = d.drop(STR_COLS).join(s, on=["q", "s1"], how="left").select(list(pl.read_parquet_schema(f)))
        d.write_parquet(a.out / "test" / "pruned" / f.name)
        log(f"recomputed {f.name}: {d.height} pairs")
    shutil.copy(a.emb, a.out / "test" / "emb_cos_pruned.parquet")
    shutil.copy(a.base / "config.json", a.out / "config.json")
    for m in a.models.glob("*.txt"):
        shutil.copy(m, a.out / m.name)

    # 3. p3 for France pairs with the saved models (group features never cross countries)
    from ber.blend import _p3
    from ber.stage2 import _load
    d = _load(a.out, "test")
    src = pl.read_parquet(a.out / "test" / "records.parquet", columns=["src"]).with_row_index("q")
    d = d.join(src.select(pl.col("q").cast(d["q"].dtype), pl.col("src").alias("qsrc")), on="q", how="left")
    fr3 = _p3(a.out, "test", d).select("q", "s1", "p3")
    old = pl.read_parquet(a.scores)
    frq = fr3.select("q").unique()
    oldfr = old.join(frq, on="q", how="inner")
    log(f"France pairs: old {oldfr.height}, new {fr3.height}")
    cmp = oldfr.join(fr3.rename({"p3": "p3_new"}), on=["q", "s1"], how="inner")
    log(f"matched pairs {cmp.height}; mean p3 old {cmp['p3'].mean():.4f} new {cmp['p3_new'].mean():.4f}; "
        f"best>0.5 old {(oldfr.group_by('q').agg(pl.col('p3').max())['p3'] > 0.5).mean():.4f} "
        f"new {(fr3.group_by('q').agg(pl.col('p3').max())['p3'] > 0.5).mean():.4f}")
    pl.concat([old.join(frq, on="q", how="anti"), fr3.select(old.columns)]).write_parquet(a.out / "scores_test.parquet")

    # 4. zone raw features: France rows replaced
    raw = pl.read_parquet(a.raw)
    zone = raw.select("q", "s1")
    emb = pl.read_parquet(a.emb).with_columns(pl.col("q").cast(d["q"].dtype), pl.col("s1").cast(d["s1"].dtype))
    newraw = d.select("q", "s1", *RAW_COLS).join(zone, on=["q", "s1"], how="inner") \
              .join(emb.select("q", "s1", "emb_cos"), on=["q", "s1"], how="left")  # as scripts/zone_raw_feats.py
    newraw = newraw.with_columns(*[pl.col(c).cast(raw.schema[c]) for c in newraw.columns])
    pl.concat([raw.join(newraw.select("q", "s1"), on=["q", "s1"], how="anti"), newraw.select(raw.columns)]) \
        .write_parquet(a.out / "raw_test.parquet")
    log(f"raw zone features: {newraw.height} France rows replaced")


if __name__ == "__main__":
    main()

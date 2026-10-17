"""Extract raw pair features (similarities, house/postal, lengths, stage-A pA) for the cross-encoder zone pairs,
so the blend stacker can combine them with cross-encoder scores.

python scripts/zone_raw_feats.py --run /work/run4 --ce /work/ce_out3 /work/ce_extra_out --out /work/rawf
writes <out>/raw_val.parquet (train split, validation zone pairs) and <out>/raw_test.parquet.
"""
import argparse
from pathlib import Path

import polars as pl

COLS = ["cos_a", "cos_n", "n_cand", "rel_top1", "gap12", "name_ratio", "name_tsr", "name_pr", "name_jw", "skel_ratio",
        "skel_tsr", "addr_ratio", "addr_tsr", "addr_pr", "street_tsr", "street_ratio", "house_sim", "house_eq",
        "house_miss", "house_absdiff", "postal_eq", "postal_miss", "num_jac", "num_inter", "first_tok_eq", "q_nonascii",
        "q_addr_empty", "q_src3", "len_name1", "len_name2", "len_addr1", "len_addr2", "pA"]

ap = argparse.ArgumentParser()
ap.add_argument("--run", type=Path, required=True)
ap.add_argument("--ce", type=Path, nargs="+", required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
for split, name in (("train", "val"), ("test", "test")):
    zone = pl.concat([pl.read_parquet(c / f"ce_scores_{name}.parquet", columns=["q", "s1"]) for c in a.ce]) \
        .with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32)).unique()
    d = pl.scan_parquet(str(a.run / split / "pruned" / "*.parquet")).select("q", "s1", *COLS) \
        .join(zone.lazy(), on=["q", "s1"], how="inner").collect()
    e = pl.scan_parquet(a.run / split / "emb_cos_pruned.parquet").join(zone.lazy(), on=["q", "s1"], how="inner").collect()
    d = d.join(e.select("q", "s1", "emb_cos"), on=["q", "s1"], how="left")
    d.write_parquet(a.out / f"raw_{name}.parquet")
    print(name, "zone", zone.height, "with features", d.height, flush=True)

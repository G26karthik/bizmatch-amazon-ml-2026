"""Convert raw TSVs to Parquet and audit labels. Run on worker.

python -m ber.prep_audit --raw /work/student_resource/dataset --out /work/data
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import polars as pl

SRC_COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path: Path) -> pl.DataFrame:
    # quote_char=None: addresses contain quotes/commas; tabs are the only separator.
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False,
                       missing_utf8_is_empty_string=True, encoding="utf8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rep: dict = {}
    t0 = time.time()

    frames = {}
    for split in ("train", "test"):
        for s in (1, 2, 3):
            name = f"{split}_source{s}"
            df = read_tsv(a.raw / split / f"{name}.tsv")
            assert df.columns == SRC_COLS, (name, df.columns)
            df.write_parquet(a.out / f"{name}.parquet")
            frames[name] = df
            rep[name] = {
                "rows": df.height,
                "dup_ids": df.height - df["entity_id"].n_unique(),
                "bad_prefix": int((~df["entity_id"].str.starts_with(f"S{s}-")).sum()),
                "countries": dict(df["country"].value_counts().iter_rows()),
                "empty_addr": int((df["business_address"].str.strip_chars() == "").sum()),
                "exact_dup_name_addr_country": df.height - df.select(SRC_COLS[1:]).n_unique(),
            }
            print(name, rep[name], f"{time.time() - t0:.0f}s", flush=True)

    gt = read_tsv(a.raw / "train" / "train_ground_truth.tsv")
    assert gt.columns == ["source1_entity_id", "matched_entity_ids"], gt.columns
    gt.write_parquet(a.out / "train_ground_truth.parquet")
    pairs = (gt.with_columns(pl.col("matched_entity_ids").str.split(","))
               .explode("matched_entity_ids")
               .rename({"source1_entity_id": "s1", "matched_entity_ids": "m"})
               .with_columns(pl.col("m").str.strip_chars())
               .filter(pl.col("m") != ""))
    pairs.write_parquet(a.out / "train_pairs.parquet")
    n_per = gt.select(pl.col("matched_entity_ids").str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.len().alias("n"))["n"]

    s1 = frames["train_source1"].select(pl.col("entity_id").alias("s1"), pl.col("country").alias("c1"))
    other = pl.concat([frames["train_source2"], frames["train_source3"]]).select(
        pl.col("entity_id").alias("m"), pl.col("country").alias("cm"))
    j = pairs.join(s1, on="s1", how="left").join(other, on="m", how="left")
    matched_ids = pairs["m"].n_unique()
    rep["gt"] = {
        "rows": gt.height,
        "dup_s1": gt.height - gt["source1_entity_id"].n_unique(),
        "s1_not_in_source1": int(j["c1"].is_null().sum()),
        "match_not_in_s2s3": int(j["cm"].is_null().sum()),
        "pairs": pairs.height,
        "pairs_s2": int(pairs["m"].str.starts_with("S2-").sum()),
        "pairs_s3": int(pairs["m"].str.starts_with("S3-").sum()),
        "dup_pair_within_row": pairs.height - pairs.unique().height,
        "matched_ids_unique": matched_ids,
        "matched_ids_shared_by_multiple_s1": pairs.height - matched_ids,
        "country_mismatch_pairs": int((j["c1"] != j["cm"]).sum()),
        "n_matches_hist": {str(k): v for k, v in pl.DataFrame({"n": n_per}).with_columns(
            pl.col("n").clip(upper_bound=20))["n"].value_counts().sort("n").iter_rows()},
        "s2s3_train_unmatched": other.height - matched_ids,
        "s2s3_train_total": other.height,
    }
    print("gt", json.dumps(rep["gt"], indent=1), flush=True)

    # Sample matched pairs for eyeballing noise patterns
    names = pl.concat([frames["train_source1"], frames["train_source2"], frames["train_source3"]])
    samp = (pairs.sample(40, seed=0)
            .join(names.rename({c: f"{c}_1" for c in SRC_COLS}).rename({"entity_id_1": "s1"}), on="s1")
            .join(names.rename({c: f"{c}_m" for c in SRC_COLS}).rename({"entity_id_m": "m"}), on="m"))
    with pl.Config(tbl_rows=40, fmt_str_lengths=60, tbl_width_chars=250):
        print(samp.select("business_name_1", "business_name_m", "business_address_1", "business_address_m", "country_1"))

    (a.out / "audit.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

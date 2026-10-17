"""Raw-name copy-proxy features for one-op pair files (same definitions as tb_build_ns.py):
rx_* = S2/S3 records whose exact name string equals the S1's name; rl_* = same on lower-cased, non-word-stripped key;
rec_lk_eq = record stripped name == S1 stripped name; s_addr_len. Writes <pairs>_raw.parquet (m, s1, features)."""
import sys

import polars as pl


def loose(col):
    return pl.col(col).fill_null("").str.to_lowercase().str.replace_all(r"[^\w]", "")


def run(split, files):
    pr = pl.concat([pl.read_parquet(f"local_data/work/{f}.parquet", columns=["m", "s1"]) for f in files]).unique()
    s1 = (pl.scan_parquet(f"local_data/{split}_source1.parquet").join(pr.lazy().select(entity_id="s1").unique(), on="entity_id", how="semi")
          .select(s1="entity_id", bn=pl.col("business_name").fill_null(""), s_addr_len=pl.col("business_address").fill_null("").str.len_chars())
          .collect().with_columns(lk=loose("bn")))
    rec = pl.concat([pl.scan_parquet(f"local_data/{split}_source{s}.parquet").join(pr.lazy().select(entity_id="m").unique(), on="entity_id", how="semi")
                     .select(m="entity_id", rlk=loose("business_name")).collect() for s in (2, 3)])
    parts = []
    for s in (2, 3):
        parts.append(pl.scan_parquet(f"local_data/{split}_source{s}.parquet")
                     .select(bn=pl.col("business_name").fill_null(""), lk=loose("business_name"),
                             emp=(pl.col("business_address").fill_null("").str.strip_chars() == ""))
                     .join(s1.lazy().select("lk").unique(), on="lk", how="semi").collect(engine="streaming").with_columns(src=pl.lit(f"S{s}")))
    rp = pl.concat(parts)
    rx = rp.group_by("bn").agg(rx_all=pl.len(), rx_s2=(pl.col("src") == "S2").sum(), rx_emp=pl.col("emp").sum())
    rl = rp.group_by("lk").agg(rl_all=pl.len(), rl_s2=(pl.col("src") == "S2").sum(), rl_emp=pl.col("emp").sum())
    s1 = s1.join(rx, on="bn", how="left").join(rl, on="lk", how="left").fill_null(0)
    out = (pr.join(s1, on="s1", how="left").join(rec, on="m", how="left")
           .with_columns(is2=pl.col("m").str.starts_with("S2"))
           .with_columns(rx_same=pl.when(pl.col("is2")).then(pl.col("rx_s2")).otherwise(pl.col("rx_all") - pl.col("rx_s2")),
                         rl_same=pl.when(pl.col("is2")).then(pl.col("rl_s2")).otherwise(pl.col("rl_all") - pl.col("rl_s2")),
                         rec_lk_eq=(pl.col("lk") == pl.col("rlk")).cast(pl.Int64))
           .select("m", "s1", "rx_all", "rx_same", "rx_emp", "rl_all", "rl_same", "rl_emp", "rec_lk_eq", "s_addr_len"))
    for f in files:
        d = pl.read_parquet(f"local_data/work/{f}.parquet", columns=["m", "s1"]).join(out, on=["m", "s1"], how="left")
        d.write_parquet(f"local_data/work/{f}_raw.parquet")
        print(split, f, d.height, d.null_count().sum_horizontal().item(), flush=True)


if __name__ == "__main__":
    sp = sys.argv[1]
    run(sp, ["tb_pairs_" + sp, "tb_big_" + sp])

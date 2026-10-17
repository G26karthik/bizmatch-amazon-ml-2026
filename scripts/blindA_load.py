"""blindA: load raw TSVs exactly (row order preserved), cache to parquet, basic profile.
Outputs /work/agent_blindA/raw_{split}_s{k}.parquet with columns row, entity_id, name, addr, country, idn
"""
import polars as pl, sys, os
D = "/work/sr/student_resource/dataset"
O = "/work/agent_blindA"
for split in ["train", "test"]:
    for k in [1, 2, 3]:
        p = f"{D}/{split}/{split}_source{k}.tsv"
        # raw read: no quoting
        df = pl.read_csv(p, separator="\t", quote_char=None, has_header=True, infer_schema=False,
                         missing_utf8_is_empty_string=True)
        df = df.rename({"business_name": "name", "business_address": "addr"})
        df = df.with_row_index("row").with_columns(
            pl.col("entity_id").str.slice(3).cast(pl.Int64).alias("idn"))
        n_lines = sum(1 for _ in open(p, "rb")) - 1
        print(split, k, df.shape, "lines", n_lines, "cols", df.columns,
              "dup ids", df.height - df["entity_id"].n_unique(),
              "idn min/max", df["idn"].min(), df["idn"].max(), flush=True)
        df.write_parquet(f"{O}/raw_{split}_s{k}.parquet")
gt = pl.read_csv(f"{D}/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema=False,
                 missing_utf8_is_empty_string=True).with_row_index("gtrow")
print("gt", gt.shape)
gt.write_parquet(f"{O}/gt_raw.parquet")

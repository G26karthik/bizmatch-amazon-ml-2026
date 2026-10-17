"""E00 part 5: why are never-retrieved empty-address copies missed, and what would recover them."""
from pathlib import Path

import polars as pl

OUT = Path("/work/agent_e00")
NB = 89043
e = pl.read_parquet(OUT / "err_records.parquet")
rr = pl.read_parquet(OUT / "retr_ranks.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
nr = e.filter(pl.col("cat") == "FN_not_retrieved").join(rr.drop("country"), on=["q", "s1"], how="left")
nr = nr.with_columns((pl.col("q_name") == pl.col("s_name")).alias("name_eq"),
                     pl.when(pl.col("q_name_cnt") == 0).then(pl.lit("0")).when(pl.col("q_name_cnt") == 1).then(pl.lit("1"))
                       .when(pl.col("q_name_cnt") <= 5).then(pl.lit("2-5")).when(pl.col("q_name_cnt") <= 50).then(pl.lit("6-50"))
                       .otherwise(pl.lit(">50")).alias("qn_cnt"),
                     (pl.col("kth_n") == 0).alias("name_retr_short"),  # fewer than K_NAME S1s above the 0.01 score floor
                     pl.when(pl.col("rank_n") < 50).then(pl.lit("<50")).when(pl.col("rank_n") < 100).then(pl.lit("50-99"))
                       .when(pl.col("rank_n") < 300).then(pl.lit("100-299")).otherwise(pl.lit(">=300/none")).alias("rank_n_b"))
cfg = pl.Config(tbl_rows=50, tbl_width_chars=220)


def g(df, by, title):
    x = df.group_by(by).agg(pl.len().alias("n"), (pl.col("loss").sum() / NB).alias("F05"),
                            ((pl.col("loss") * pl.col("acc").fill_null(0)).sum() / NB).alias("F05_rec_est"),
                            pl.col("cos_n").median().alias("cos_n_med"), pl.col("in_e5").mean().alias("e5")).sort("F05", descending=True)
    print(f"\n--- {title} ---")
    with cfg:
        print(x)


emp = nr.filter(pl.col("q_addr_empty"))
g(emp, ["name_eq", "qn_cnt"], "never-retrieved empty-address: record name == S1 name x #pool S1s with the record's exact name")
g(emp, ["rank_n_b"], "never-retrieved empty-address: rank of the true S1 in the (DF-capped) name retriever")
g(emp, ["name_retr_short"], "never-retrieved empty-address: name retriever returned < 8 S1s above score floor")
g(emp, ["recov", "psc_bin"], "never-retrieved empty-address: recoverability x parent similarity")
g(nr.filter(~pl.col("q_addr_empty")), ["recov"], "never-retrieved WITH address: recoverability")
# exact-name blocking: how many never-retrieved copies share the record's exact name with the parent, by pool count
for K in (1, 3, 10, 50):
    d = emp.filter(pl.col("name_eq") & (pl.col("q_name_cnt") <= K))
    print(f"exact-name blocking (record name == S1 name, <= {K} pool S1s with the name): {d.height} copies, loss {d['loss'].sum() / NB:.6f}, "
          f"analogue-recoverable {(d['loss'] * d['acc'].fill_null(0)).sum() / NB:.6f}")

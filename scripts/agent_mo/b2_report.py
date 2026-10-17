"""agent_mo task B report: per-class train precision, test counts by fr27 status, decoy-excess check.

uv run --no-project --with polars==1.44.2 python scripts/agent_mo/b2_report.py
"""
import sys

import polars as pl

pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(300)
TR_FRAC = 0.5
N1 = pl.DataFrame({"country": ["US", "India"], "S1tr": [1323633, 883188], "S1te": [663106, 809986]})
tr = pl.concat([pl.read_parquet(f"local_data/work/agent_mo_B_train_{c}.parquet") for c in ("US", "India") if __import__("os").path.exists(f"local_data/work/agent_mo_B_train_{c}.parquet") and __import__("os").path.exists(f"local_data/work/agent_mo_B_test_{c}.parquet")], how="diagonal_relaxed")
te = pl.concat([pl.read_parquet(f"local_data/work/agent_mo_B_test_{c}.parquet") for c in ("US", "India") if __import__("os").path.exists(f"local_data/work/agent_mo_B_train_{c}.parquet") and __import__("os").path.exists(f"local_data/work/agent_mo_B_test_{c}.parquet")], how="diagonal_relaxed")


def feats(df):
    first = lambda c: pl.col(c).str.to_lowercase().str.extract(r"(\w+)")
    return df.with_columns(pl.col("m").str.slice(0, 2).alias("src"), pl.col("ops").clip(2, 4).alias("opsc"), pl.col("nrare").clip(2, 4).alias("nrarec"),
                           (first("business_name") == first("name1")).alias("first_same"),
                           (pl.col("nsh") / pl.max_horizontal("nr", "ns")).round(1).alias("shfrac"))


tr, te = feats(tr), feats(te)


def table(gb):
    a = tr.group_by(gb).agg(pl.len().alias("tr_n"), pl.col("right").sum().alias("tr_right"))
    b = te.group_by(gb).agg(pl.len().alias("te_n"), (pl.col("status") == "acc_same").sum().alias("te_acc"),
                            (pl.col("status") == "acc_other").sum().alias("te_oth"), (pl.col("status") == "unassigned").sum().alias("te_un"))
    x = a.join(b, on=gb, how="full", coalesce=True).fill_null(0).join(N1, on="country")
    x = x.with_columns((pl.col("tr_right") / pl.col("tr_n")).round(4).alias("P_train"),
                       (pl.col("tr_n") / TR_FRAC / pl.col("S1tr") * 1000).round(3).alias("tr_all_1k"),
                       (pl.col("tr_right") / TR_FRAC / pl.col("S1tr") * 1000).round(3).alias("tr_copy_1k"),
                       (pl.col("te_n") / pl.col("S1te") * 1000).round(3).alias("te_all_1k"),
                       (pl.col("te_acc") / pl.col("S1te") * 1000).round(3).alias("te_acc_1k"),
                       (pl.col("te_un") / pl.col("S1te") * 1000).round(3).alias("te_un_1k"))
    x = x.with_columns((pl.col("te_all_1k") / (pl.col("tr_all_1k") + 1e-9)).round(3).alias("all_ratio"),
                       ((pl.col("tr_copy_1k") - pl.col("te_acc_1k")) / (pl.col("te_un_1k") + 1e-9)).round(3).alias("P_impl_un"))
    return x.drop("S1tr", "S1te").sort(gb)


for gb in (["country"], ["country", "src"], ["country", "src", "opsc"], ["country", "opsc"], ["country", "opsc", "nrarec"], ["country", "first_same"], ["country", "opsc", "first_same"], ["country", "shfrac"]):
    print(table(gb))
if len(sys.argv) > 1:
    for c in ("US", "India"):
        print(tr.filter((pl.col("country") == c) & ~pl.col("right")).select("business_name", "name1", "business_address", "addr1", "ops", "nrare").sample(12, seed=1))
        print(tr.filter((pl.col("country") == c) & pl.col("right")).select("business_name", "name1", "business_address", "addr1", "ops", "nrare").sample(8, seed=1))
        print(te.filter((pl.col("country") == c) & (pl.col("status") == "unassigned")).select("business_name", "name1", "business_address", "addr1", "ops", "nrare").sample(12, seed=1))

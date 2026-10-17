"""Calibration: marginal precision per p-bin on train OOF; p distribution of test top picks (unassigned vs fr27-assigned)."""
import polars as pl
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_width_chars(250)
t = pl.read_parquet("local_data/work/tb_oof_train_top.parquet")
tt = pl.read_parquet("local_data/work/tb_test_top.parquet")
bins = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]
def b(d):
    return d.with_columns(bin=pl.col("p").cut(bins, left_closed=True))
print("TRAIN OOF marginal precision per p-bin (by country):")
print(b(t).group_by("country", "bin").agg(pl.len(), pl.col("p").mean().round(3).alias("mean_p"), pl.col("label").mean().round(4).alias("prec")).sort("country", "bin"))
print("TEST top-pick p-bins: unassigned vs fr27-assigned (agree w/ fr27 for assigned):")
tt = tt.with_columns(asg=pl.col("fr27_s1").is_not_null(), agree=pl.col("s1") == pl.col("fr27_s1"))
print(b(tt).group_by("country", "asg", "bin").agg(pl.len(), pl.col("p").mean().round(3).alias("mean_p"), pl.col("agree").mean().round(4).alias("agree_fr27")).sort("country", "asg", "bin"))
print("TRAIN OOF p-bin share vs TEST unassigned share:")
for c in ("US", "India", "France"):
    for name, d in (("train", t.filter(pl.col("country") == c)), ("test_un", tt.filter((pl.col("country") == c) & ~pl.col("asg")))):
        if d.height:
            print(c, name, d.height, "share p>=0.8/0.85/0.9:", [round((d["p"] >= x).mean(), 3) for x in (0.8, 0.85, 0.9)], "exp prec among p>=0.85:", round(d.filter(pl.col("p") >= 0.85)["p"].mean() or 0, 4))

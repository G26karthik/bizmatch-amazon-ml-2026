"""Can the weak namesake cues resolve empty-address namesakes confidently? Fit a small LightGBM on half-A namesake
candidate rows (cues + current model probability), normalise within each record, and on half B report top-1
accuracy and how many records reach a confidence where selecting the match pays off under F0.5 (~>0.75).
Needs /work/dive/namesakes_{A,B}_pA.parquet from scripts/dive_namesakes.py."""
import numpy as np
import polars as pl
import lightgbm as lgb

F = ["prob", "p3", "raw_ratio", "raw_eq", "bare_eq", "suf_eq", "case_eq", "n_copies", "copy_max_ratio", "copy_mean_ratio",
     "copy_raw_eq", "copy_bare_eq", "copy_suf_share", "copy_case_share", "copy_same_src", "k"]
def X(d): return d.select([pl.col(c).cast(pl.Float32) for c in F]).to_numpy()
A = pl.read_parquet("/work/dive/namesakes_A_pA.parquet"); B = pl.read_parquet("/work/dive/namesakes_B_pA.parquet")
m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=50, verbose=-1, num_threads=2, seed=1),
              lgb.Dataset(X(A), A["y"].to_numpy()), 300)
for name, d in (("half B", B),):
    d = d.with_columns(pl.Series("s", m.predict(X(d)))).with_columns((pl.col("s") / pl.col("s").sum().over("q")).alias("sn"),
                                                                        (pl.col("prob") / pl.col("prob").sum().over("q")).alias("pn"))
    for col in ("pn", "sn"):
        top = d.sort(col, descending=True).group_by("q", maintain_order=True).first()
        print(f"{name} {col}: records {top.height}, top-1 acc {top['y'].mean():.3f}")
        for thr in (0.6, 0.75, 0.9):
            t = top.filter(pl.col(col) >= thr)
            print(f"   confidence>={thr}: {t.height} records ({t.height / top.height:.3f}), precision {t['y'].mean() if t.height else float('nan'):.3f}")
imp = sorted(zip(F, m.feature_importance("gain")), key=lambda x: -x[1])[:8]
print("importance:", [(f, int(g)) for f, g in imp])

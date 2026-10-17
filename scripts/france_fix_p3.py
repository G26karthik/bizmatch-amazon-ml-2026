"""Stack probabilities (p3) for France test pairs from France-fixed features (scripts/france_fix.py output), with the
models of another stack run; other countries keep that run's p3.

France-fixed features parse "N°24"-style house numbers (23k France records had none) and drop region/department and
abbreviation noise (street similarity >= 90 for 1.60M France pairs instead of 1.17M), which the base features score as
mismatches. Writes <out>/scores_test.parquet and links <out>/scores_val.parquet, train/, test/ and the models.

python scripts/france_fix_p3.py --fixed /work/runfr --run /work/run28 --out /work/runfr28
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, "src")
from ber.blend import _p3  # noqa: E402
from ber.pipeline import log  # noqa: E402
from ber.stage2 import _load  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--fixed", type=Path, required=True, help="france_fix.py output dir (test/records, test/pruned France_*)")
ap.add_argument("--run", type=Path, required=True, help="stack run whose models and scores are used")
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
for m in ("s1_fold0.txt", "s1_fold1.txt", "s2_fold0.txt", "s2_fold1.txt", "s3_model.txt", "config.json"):
    shutil.copy(a.run / m, a.out / m)
for name, target in (("test", a.fixed / "test"), ("train", (a.run / "train").resolve()), ("scores_val.parquet", a.run / "scores_val.parquet")):
    p = a.out / name
    if p.is_symlink() or p.exists():
        p.unlink()
    os.symlink(target, p)
d = _load(a.out, "test")  # France pairs only (the fixed dir holds France parts)
src = pl.read_parquet(a.out / "test" / "records.parquet", columns=["src"]).with_row_index("q")
d = d.join(src.select(pl.col("q").cast(d["q"].dtype), pl.col("src").alias("qsrc")), on="q", how="left")
fr3 = _p3(a.out, "test", d).select("q", "s1", "p3")
old = pl.read_parquet(a.run / "scores_test.parquet")
frq = fr3.select("q").unique()
oldfr = old.join(frq, on="q", how="inner")
cmp = oldfr.join(fr3.rename({"p3": "p3_new"}), on=["q", "s1"], how="inner")
log(f"France pairs {fr3.height} (old {oldfr.height}); mean p3 {cmp['p3'].mean():.4f} -> {cmp['p3_new'].mean():.4f}; "
    f"records with best p3 > 0.5: {(oldfr.group_by('q').agg(pl.col('p3').max())['p3'] > 0.5).mean():.4f} -> "
    f"{(fr3.group_by('q').agg(pl.col('p3').max())['p3'] > 0.5).mean():.4f}")
pl.concat([old.join(frq, on="q", how="anti"), fr3.select(old.columns)]).write_parquet(a.out / "scores_test.parquet")
log("wrote scores_test.parquet")

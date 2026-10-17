"""E00 part 9: model-free precision of 'empty-address record whose exact name matches exactly one pool S1' on half B.
Shows whether the empty-address recall bucket is separable at all (copy vs hidden-S1 copy vs unmatched record)."""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, "src")
from ber.pipeline import SEED, hide_pct, hidden_s1, labels  # noqa: E402
from ber.stage2 import _chosen  # noqa: E402

work = Path("/work/run28")
pct = hide_pct(work)
lab = labels(Path("/work/data"), work).with_columns((pl.col("parent").fill_null(pl.col("rid")).hash(SEED + 7) % 2 == 0).alias("half_a"))
SB = lab.filter((pl.col("src") == 1) & pl.col("is_val") & ~pl.col("half_a")).select(pl.col("rid").cast(pl.UInt32).alias("s1"))
rec = pl.read_parquet(work / "train" / "records.parquet", columns=["entity_id", "country", "src", "name_n", "addr_empty"]).with_row_index("rid") \
        .with_columns(pl.col("rid").cast(pl.UInt32))
pool = rec.filter((pl.col("src") == 1) & ~hidden_s1(pl.col("entity_id"), pct))
uniq = pool.group_by("country", "name_n").agg(pl.len().alias("n"), pl.col("rid").first().alias("s1")).filter(pl.col("n") == 1).drop("n")
gt = pl.read_parquet("/work/data/train_pairs.parquet")
H = rec.join(gt.filter(hidden_s1(pl.col("s1"), pct)).select(pl.col("m").alias("entity_id")), on="entity_id", how="semi").select("rid")
q = rec.filter((pl.col("src") != 1) & pl.col("addr_empty")).join(uniq, on=["country", "name_n"]).join(SB, on="s1", how="semi")
q = q.join(lab.select(pl.col("rid").cast(pl.UInt32), pl.col("parent").cast(pl.UInt32)), on="rid", how="left")
q = q.with_columns(pl.col("rid").is_in(H["rid"].implode()).alias("isH"))
q = q.with_columns(pl.when(pl.col("parent") == pl.col("s1")).then(pl.lit("copy of this S1"))
                     .when(pl.col("parent").is_not_null()).then(pl.lit("copy of another S1"))
                     .when(pl.col("isH")).then(pl.lit("copy of a HIDDEN S1 (simulation orphan)"))
                     .otherwise(pl.lit("unmatched record")).alias("truth"))
vb = pl.read_parquet(work / "val_best_run28_7q.parquet").with_columns(pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
ch = _chosen(vb, {"mode": "efopt", "miss": 0.0, "gamma": 1.5}).select(pl.col("q").cast(pl.UInt32).alias("rid"), pl.col("s1").cast(pl.UInt32).alias("ch"))
q = q.join(ch, on="rid", how="left").with_columns((pl.col("ch") == pl.col("s1")).fill_null(False).alias("model_chose"))
print(f"half-B S1s with a unique pool name, empty-address records carrying exactly that name: {q.height}")
with pl.Config(tbl_rows=20, tbl_width_chars=200, fmt_str_lengths=50):
    print(q.group_by("truth").agg(pl.len(), pl.col("model_chose").mean().alias("model_chose_rate")).with_columns(
        (pl.col("len") / q.height).alias("share")).sort("len", descending=True))
print(f"precision of the rule 'accept all': {(q['truth'] == 'copy of this S1').mean():.3f}; without simulation orphans: "
      f"{q.filter(pl.col('truth') != 'copy of a HIDDEN S1 (simulation orphan)')['truth'].eq('copy of this S1').mean():.3f}")

"""Build cross-encoder datasets (raw text pairs) for the uncertain zone.

Zone: each record's top-3 candidates by stage-A probability, restricted to pA < 0.98 (confident pairs need no help).
  ce_train.parquet   non-validation training pairs (labels), sampled
  ce_val.parquet     validation pairs (labels kept for blending/evaluation only)
  ce_test.parquet    test pairs
Text is the raw "name | address" (native scripts kept; the multilingual model reads them directly).

python -m ber.ce_data --data /work/data --work /work/run6 --out /work/ce --n_train 700000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from ber.pipeline import SEED, labels, log

ZONE_MAX_PA, TOP_R = 0.98, 3


def _zone(work: Path, split: str, top_r: int = TOP_R, max_pa: float = ZONE_MAX_PA) -> pl.DataFrame:
    from ber.stage2 import _load
    d = _load(work, split).select("q", "s1", "pA")
    d = d.with_columns(pl.col("pA").rank("ordinal", descending=True).over("q").alias("r"))
    return d.filter((pl.col("r") <= top_r) & (pl.col("pA") < max_pa)).drop("r")


def build_extra(data: Path, work: Path, out: Path, top_r: int, max_pa: float) -> None:
    """Pairs of the wider zone that are not in the default zone (to be scored by already-trained cross-encoders)."""
    from ber.blend import eval_queries
    out.mkdir(parents=True, exist_ok=True)
    ev = eval_queries(data, work).implode()
    for split, name in (("train", "val"), ("test", "test")):
        t = _texts(work, split)
        z = _zone(work, split, top_r, max_pa).join(_zone(work, split), on=["q", "s1"], how="anti")
        if split == "train":
            z = z.filter(pl.col("q").is_in(ev))
        z = z.join(t.rename({"rid": "s1", "t": "text_a"}), on="s1").join(t.rename({"rid": "q", "t": "text_b"}), on="q")
        z.select("q", "s1", "text_a", "text_b").write_parquet(out / f"ce_{name}.parquet")
        log(f"extra zone {name}: {z.height} pairs")


def _texts(work: Path, split: str) -> pl.DataFrame:
    rec = pl.read_parquet(work / split / "records.parquet", columns=["business_name", "business_address"]).with_row_index("rid")
    return rec.select("rid", (pl.col("business_name") + " | " + pl.col("business_address")).alias("t"))


def build(data: Path, work: Path, out: Path, n_train: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    lab = labels(data, work).select(pl.col("rid").alias("q"), "parent", "is_val")
    t = _texts(work, "train")
    z = _zone(work, "train").join(lab, on="q", how="left")
    z = z.with_columns((pl.col("parent") == pl.col("s1")).fill_null(False).cast(pl.Int8).alias("y"))
    z = z.join(t.rename({"rid": "s1", "t": "text_a"}), on="s1").join(t.rename({"rid": "q", "t": "text_b"}), on="q")
    from ber.blend import eval_queries
    ev = eval_queries(data, work).implode()
    # Training pairs: records that never touch a validation S1 (and are not validation records themselves), so
    # every pair scored for validation is out-of-sample for the cross-encoder.
    tr = z.filter(~pl.col("is_val") & ~pl.col("q").is_in(ev))
    tr = tr.sample(min(n_train, tr.height), seed=SEED)
    tr.select("text_a", "text_b", "y").write_parquet(out / "ce_train.parquet")
    va = z.filter(pl.col("q").is_in(ev))
    va.select("q", "s1", "text_a", "text_b", "y").write_parquet(out / "ce_val.parquet")
    log(f"ce train {tr.height} (pos {int(tr['y'].sum())}), eval pairs {va.height}")
    t = _texts(work, "test")
    zt = _zone(work, "test").join(t.rename({"rid": "s1", "t": "text_a"}), on="s1").join(t.rename({"rid": "q", "t": "text_b"}), on="q")
    zt.select("q", "s1", "text_a", "text_b").write_parquet(out / "ce_test.parquet")
    log(f"ce test {zt.height}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/work/data"))
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("/work/ce"))
    ap.add_argument("--n_train", type=int, default=700_000)
    ap.add_argument("--extra", action="store_true", help="only the wider-zone pairs outside the default zone")
    ap.add_argument("--top_r", type=int, default=5)
    ap.add_argument("--max_pa", type=float, default=0.9999)
    a = ap.parse_args()
    if a.extra:
        build_extra(a.data, a.work, a.out, a.top_r, a.max_pa)
    else:
        build(a.data, a.work, a.out, a.n_train)

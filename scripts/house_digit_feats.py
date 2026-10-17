"""House-number digit-pattern features for zone pairs (added to the stacker's raw features).

Generator rules (training labels): decoys shift the house number by 1-50 and keep its digit count (same-length
number with name and street matching: 78% decoy), while copy noise deletes or substitutes a digit (a number one digit
shorter: 86% true copy). The tree stack sees house equality, a digit edit similarity and log |difference| but not
these patterns explicitly.
Columns: hd_len_diff (len(q) - len(s1)), hd_same_len, hd_one_deleted, hd_one_substituted, hd_offset_small_same_len,
hd_transposed.

python scripts/house_digit_feats.py --base /work/base --raw /work/rawf --out /work/rawf4
"""
import argparse
from pathlib import Path

import polars as pl


def pattern(a: str, b: str) -> tuple[int, int, int, int, int, int]:
    """a = record house, b = S1 house (digit strings); returns the six features (all 0 when either is missing)."""
    if not a or not b or not a.isdigit() or not b.isdigit() or a == b:
        return (0, 0, 0, 0, 0, 0)
    la, lb = len(a), len(b)
    one_del = int(la == lb - 1 and any(b[:i] + b[i + 1:] == a for i in range(lb)))
    same = int(la == lb)
    subs = int(same and sum(x != y for x, y in zip(a, b)) == 1)
    trans = int(same and sorted(a) == sorted(b) and sum(x != y for x, y in zip(a, b)) == 2)
    off = int(same and not subs and abs(int(a) - int(b)) <= 50)
    return (la - lb, same, one_del, subs, off, trans)


ap = argparse.ArgumentParser()
ap.add_argument("--base", type=Path, required=True)
ap.add_argument("--raw", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
names = ["hd_len_diff", "hd_same_len", "hd_one_deleted", "hd_one_substituted", "hd_offset_small_same_len", "hd_transposed"]
for split, name in (("train", "val"), ("test", "test")):
    rec = pl.read_parquet(a.base / split / "records.parquet", columns=["house"]).with_row_index("rid") \
            .with_columns(pl.col("rid").cast(pl.UInt32), pl.col("house").fill_null(""))
    r = pl.read_parquet(a.raw / f"raw_{name}.parquet")
    d = r.select("q", "s1").join(rec.rename({"rid": "q", "house": "hq"}), on="q", how="left") \
         .join(rec.rename({"rid": "s1", "house": "hs"}), on="s1", how="left")
    f = [pattern(x or "", y or "") for x, y in zip(d["hq"].to_list(), d["hs"].to_list())]
    d = d.select("q", "s1").with_columns(*[pl.Series(n, [t[i] for t in f], dtype=pl.Int8) for i, n in enumerate(names)])
    r.join(d, on=["q", "s1"], how="left").write_parquet(a.out / f"raw_{name}.parquet")
    print(name, r.height, "pairs;", {n: round(float(d[n].cast(pl.Float64).abs().gt(0).mean()), 4) for n in names}, flush=True)

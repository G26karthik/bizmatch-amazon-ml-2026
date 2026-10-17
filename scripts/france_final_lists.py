"""Final France edit lists (fr13 -> fr14), merging the outputs of fr13_build_lists.py and the E01 scripts:
  /work/lead/fr13_remove_all.parquet = fr13_remove.parquet (kept vocabulary swaps) + fr_other_vocab_swaps.parquet (vocab)
  /work/lead/fr14_adds.parquet       = E01 non-word restores not in fr13_restore.parquet (legal form kept, not lowercase)
                                       + E01 address-parse-gap adds (fr_parse_adds.parquet)

python scripts/france_final_lists.py
"""
import re
import unicodedata

import polars as pl

LEGF = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei"}


def toks(s):
    return set(re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()))


u32 = [pl.col("q").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32)]
y = pl.read_parquet("/work/lead/fr_other_vocab_swaps.parquet").filter(pl.col("vocab")).select(*u32)
r = pl.read_parquet("/work/lead/fr13_remove.parquet").select(*u32)
rm = pl.concat([r, y]).unique()
rm.write_parquet("/work/lead/fr13_remove_all.parquet")
print("fr13_remove_all", rm.height)

nm = pl.read_parquet("/work/runfr/test/records.parquet", columns=["business_name"])["business_name"].to_list()
e = pl.read_parquet("/work/agent_e01/fr_nonword_restore.parquet").select(*u32)
mine = pl.read_parquet("/work/lead/fr13_restore.parquet").select(*u32)
x = e.join(mine, on=["q", "s1"], how="anti")
keep = [(toks(nm[q]) & LEGF) == (toks(nm[s]) & LEGF) and nm[q] != nm[q].lower() for q, s in zip(x["q"].to_list(), x["s1"].to_list())]
x = x.filter(pl.Series(keep))
a = pl.read_parquet("/work/agent_e01/fr_parse_adds.parquet").select(*u32)
adds = pl.concat([x, a]).unique()
adds.write_parquet("/work/lead/fr14_adds.parquet")
print("fr14_adds", adds.height, "(restores", x.height, "+ parse adds", a.height, ")")

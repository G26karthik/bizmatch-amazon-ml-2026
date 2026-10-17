"""Legal-form resolver for empty-address name collisions, measured on train truth (US/India).
For every empty-address S2/S3 record whose name key (legal/stop words removed) equals 2+ S1 names in its country,
each candidate S1 gets a legal relation to the record (canonical legal forms: ltd=limited, pvt=private, inc=incorporated,
corp=corporation, co=company, llc, lp, llp, pc, pllc, plc, ...): same / record drops / record adds / swap / none.
Rules: R_same = the unique candidate with the same legal set (and not all candidates same); R_raw = the unique candidate
whose raw name equals the record's ignoring case, spaces and punctuation. Precision (picked = true parent), coverage,
and share of records without any parent (orphans) that a rule would wrongly assign.

uv run --no-project --with polars python scripts/coll_rule.py
"""
import re
import unicodedata

import polars as pl

pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(220); pl.Config.set_tbl_cols(-1)
CANON = {"ltd": "limited", "limited": "limited", "pvt": "private", "private": "private", "inc": "inc", "incorporated": "inc",
         "corp": "corp", "corporation": "corp", "co": "co", "company": "co", "llc": "llc", "lp": "lp", "llp": "llp", "pc": "pc",
         "pllc": "pllc", "plc": "plc", "sa": "sa", "sas": "sas", "sasu": "sasu", "sarl": "sarl", "eurl": "eurl", "sci": "sci",
         "snc": "snc", "ei": "ei"}
STOP = {"the", "of", "and", "de", "des", "du", "la", "le", "les", "et", "s", "a", "d", "l"}


def toks(s):
    t = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower().replace(".", "")
    return re.findall(r"[a-z0-9]+", t)


def key(s):
    return " ".join(sorted(set(toks(s)) - set(CANON) - STOP))


def legal(s):
    return frozenset(CANON[t] for t in toks(s) if t in CANON)


def rawc(s):  # raw name without case, whitespace and punctuation (accents kept)
    return re.sub(r"[\W_]+", "", (s or "").lower())


D = "local_data"
s1 = pl.read_parquet(f"{D}/train_source1.parquet").select("entity_id", "business_name", "country")
oth = pl.concat([pl.read_parquet(f"{D}/train_source2.parquet"), pl.read_parquet(f"{D}/train_source3.parquet")], how="vertical_relaxed")
emp = oth.filter(pl.col("business_address").is_null() | (pl.col("business_address").str.strip_chars() == "")).select("entity_id", "business_name", "country")
gt = pl.read_parquet(f"{D}/train_pairs.parquet").rename({"m": "entity_id", "s1": "parent"})
emp = emp.join(gt, on="entity_id", how="left")
s1 = s1.with_columns(pl.col("business_name").map_elements(key, return_dtype=pl.String).alias("k"))
emp = emp.with_columns(pl.col("business_name").map_elements(key, return_dtype=pl.String).alias("k"))
grp = s1.group_by("country", "k").agg(pl.col("entity_id").alias("cands"), pl.col("business_name").alias("cnames"), pl.len().alias("n"))
x = emp.join(grp, on=["country", "k"], how="inner").filter(pl.col("n") >= 2)
print("empty-address records:", emp.height, " with a 2+ S1 name collision:", x.height)
rows = []
for eid, nm, par, cands, cnames, n, c in x.select("entity_id", "business_name", "parent", "cands", "cnames", "n", "country").iter_rows():
    lr, rr = legal(nm), rawc(nm)
    rel = []
    for cn in cnames:
        ls = legal(cn)
        rel.append("same" if ls == lr else ("drop" if lr < ls else ("add" if ls < lr else "swap")))
    same = [i for i, r in enumerate(rel) if r == "same"]
    raw = [i for i, cn in enumerate(cnames) if rawc(cn) == rr]
    in_grp = par in cands
    true_rel = rel[cands.index(par)] if in_grp else None
    r_same = cands[same[0]] if len(same) == 1 else None
    r_raw = cands[raw[0]] if len(raw) == 1 else None
    # combined: same-legal unique, else unique drop/add among no-same
    comp = None
    if len(same) == 1:
        comp = cands[same[0]]
    elif not same:
        da = [i for i, r in enumerate(rel) if r in ("drop", "add")]
        if len(da) == 1:
            comp = cands[da[0]]
    rows.append((c, n, par is not None, in_grp, true_rel, r_same, r_same == par if r_same else None, r_raw, r_raw == par if r_raw else None,
                 comp, comp == par if comp else None, len(same), len(set(rawc(cn) for cn in cnames)) == 1))
y = pl.DataFrame(rows, schema=["country", "n", "has_parent", "parent_in_group", "true_rel", "r_same", "same_ok", "r_raw", "raw_ok", "comp",
                                "comp_ok", "n_same", "all_raw_identical"], orient="row")
print("\nrecords by parent status:", y.group_by("has_parent", "parent_in_group").len().sort("len", descending=True).rows())
print("legal relation of the TRUE parent:", y.filter(pl.col("parent_in_group")).group_by("true_rel").len().sort("len", descending=True).rows())
print("all candidate raw names identical (unresolvable):", y["all_raw_identical"].mean())
for rule, pick, ok in (("R_same", "r_same", "same_ok"), ("R_raw", "r_raw", "raw_ok"), ("R_comp", "comp", "comp_ok")):
    z = y.filter(pl.col(pick).is_not_null())
    print(f"\n{rule}: fires on {z.height} of {y.height} ({z.height / y.height:.1%}); precision {z[ok].mean():.4f}; "
          f"fires on orphans {z.filter(~pl.col('has_parent')).height}; wrong S1 {z.filter(pl.col('has_parent') & ~pl.col(ok)).height}")
    print(z.group_by("country", pl.col("n").clip(2, 6).alias("n")).agg(pl.len(), pl.col(ok).mean().round(4).alias("prec")).sort("country", "n").rows())
y.write_parquet("local_data/work/coll_rule.parquet")

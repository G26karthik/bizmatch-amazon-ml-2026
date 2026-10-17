"""blindA (local): short one-token names (<=4 letters: acronyms / initials) and pseudo one-token names at a unique
(street key, house) S1 address.  France test (submission) vs US train (truth) / US test (submission).
Match types: token == S1's first token; == initials of S1 words (all / without legal / without stopwords); none."""
import polars as pl, sys, os
import blindA_lib_local as L
R = sys.argv[1]; C = sys.argv[2]; SUB = sys.argv[3]
L.R = R
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(50)
LEG = ["sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "ei", "inc", "llc", "ltd", "corp", "co", "lp", "llp", "pc", "pllc", "limited", "private", "pvt", "company", "corporation", "incorporated"]
STOPW = ["du", "de", "des", "la", "le", "les", "l", "d", "et", "the", "of", "and", "&"]
ACC = {"é": "e", "è": "e", "ê": "e", "à": "a", "â": "a", "î": "i", "ô": "o", "ù": "u", "û": "u", "ç": "c", "É": "E"}
def toks(c):
    e = pl.col(c).str.to_lowercase()
    for k, v in ACC.items():
        e = e.str.replace_all(k, v, literal=True)
    return e.str.replace_all(r"[^a-z0-9 ]", " ").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
def feats(j):
    j = j.with_columns(toks("name").alias("rt"), toks("s1_name").alias("st"))
    j = j.filter(pl.col("rt").list.len() == 1).with_columns(pl.col("rt").list.first().alias("r"))
    ini = lambda lst: lst.list.eval(pl.element().str.slice(0, 1)).list.join("")
    j = j.with_columns(
        (pl.col("r") == pl.col("st").list.first()).alias("m_first"),
        (pl.col("r") == ini(pl.col("st"))).alias("m_ini_all"),
        (pl.col("r") == ini(pl.col("st").list.eval(pl.element().filter(~pl.element().is_in(LEG))))).alias("m_ini_nolegal"),
        (pl.col("r") == ini(pl.col("st").list.eval(pl.element().filter(~pl.element().is_in(LEG + STOPW))))).alias("m_ini_nostop"),
        pl.col("r").is_in(pl.col("st")).alias("m_anytok"),
        (pl.col("r").str.len_chars() <= 4).alias("short"))
    j = j.with_columns(pl.when(pl.col("m_first")).then(pl.lit("first_tok")).when(pl.col("m_ini_all") | pl.col("m_ini_nolegal") | pl.col("m_ini_nostop")).then(pl.lit("initials"))
                       .when(pl.col("m_anytok")).then(pl.lit("other_tok")).otherwise(pl.lit("none")).alias("mt"))
    return j
out = []
# France test
s1 = pl.read_parquet(f"{C}/test_FR_s1.parquet").select(pl.col("entity_id").alias("s1"), pl.col("name").alias("s1_name"), "K", "h")
kh = s1.group_by(["K", "h"]).len().rename({"len": "nKH"})
s1 = s1.join(kh, on=["K", "h"]).filter(pl.col("nKH") == 1)
recs = pl.read_parquet(f"{C}/test_FR_recs.parquet").select("entity_id", "name", "K", "h").filter(pl.col("h").is_not_null())
sub = pl.read_csv(SUB, separator="\t", quote_char=None, infer_schema=False).fill_null("")
sp = sub.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "ref", "matched_entity_ids": "entity_id"})
j = feats(recs.join(s1, on=["K", "h"])).join(sp, on="entity_id", how="left")
fr = j.with_columns((pl.col("ref") == pl.col("s1")).fill_null(False).alias("to_this"), pl.col("ref").is_not_null().alias("any"))
print("FRANCE test, one-token records at a unique-address S1: assigned (to this S1 / anywhere)")
print(fr.group_by(["short", "mt"]).agg(pl.len(), pl.col("to_this").mean().round(3), pl.col("any").mean().round(3)).sort(["short", "mt"]))
# US train truth, S1 alone on key -> unique address
tr_s1 = pl.read_parquet(f"{C}/us_s1.parquet").select(pl.col("entity_id").alias("s1"), pl.col("name").alias("s1_name"), "K", "h")
kh = tr_s1.group_by(["K", "h"]).len().rename({"len": "nKH"})
tr_s1 = tr_s1.join(kh, on=["K", "h"]).filter((pl.col("nKH") == 1) & pl.col("h").is_not_null() & (pl.col("K") != ""))
tr_recs = pl.read_parquet(f"{C}/us_recs.parquet").select("entity_id", "name", "K", "h", "true_s1").filter(pl.col("h").is_not_null())
t = feats(tr_recs.join(tr_s1, on=["K", "h"]))
t = t.with_columns((pl.col("true_s1") == pl.col("s1")).fill_null(False).alias("to_this"), pl.col("true_s1").is_not_null().alias("any"))
print("\nUS TRAIN truth, one-token records at a unique-address S1: true (to this S1 / anywhere)")
print(t.group_by(["short", "mt"]).agg(pl.len(), pl.col("to_this").mean().round(3), pl.col("any").mean().round(3)).sort(["short", "mt"]))
ex = fr.filter(~pl.col("any") & pl.col("mt").is_in(["first_tok", "initials"]))
print("\nFrance unassigned one-token records matching the unique-address S1 (first token / initials):", ex.height, ex.group_by(["short", "mt"]).len().to_dicts())
print(ex.select("name", "s1_name", "mt").head(30))
ex.select("entity_id", "s1", "name", "s1_name", "mt", "short").write_parquet(f"{C}/fr_onetoken_unassigned_match.parquet")

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

s1 = pl.read_parquet(f"{C}/test_FR_s1.parquet").select(pl.col("entity_id").alias("s1"), pl.col("name").alias("s1_name"), "K", "h")
kh = s1.group_by(["K", "h"]).len().rename({"len": "nKH"})
s1 = s1.join(kh, on=["K", "h"]).filter(pl.col("nKH") == 1)
recs = pl.read_parquet(f"{C}/test_FR_recs.parquet").select("entity_id", "name", "K", "h").filter(pl.col("h").is_not_null())
sub = pl.read_csv(SUB, separator="	", quote_char=None, infer_schema=False).fill_null("")
sp = sub.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "ref", "matched_entity_ids": "entity_id"}).with_columns(pl.col("entity_id").str.slice(0, 2).alias("src"))
j = feats(recs.join(s1, on=["K", "h"])).join(sp.select("entity_id", "ref"), on="entity_id", how="left")
j = j.with_columns(pl.col("entity_id").str.slice(0, 2).alias("src"), pl.col("ref").is_not_null().alias("a"))
voc = s1.select(toks("s1_name").alias("t")).explode("t").drop_nulls().unique()["t"].to_list()
j = j.with_columns(pl.col("r").is_in(voc).alias("in_vocab"))
print("one-token 'none' non-short records: in French S1 vocabulary share:", j.filter((pl.col("mt") == "none") & ~pl.col("short"))["in_vocab"].mean())
for cls in ["initials", "none", "brand", "vocab"]:
    if cls == "initials":
        x = j.filter((pl.col("mt") == cls) & pl.col("short"))
    elif cls == "none":
        x = j.filter((pl.col("mt") == "none") & ~pl.col("short"))
    elif cls == "brand":
        x = j.filter((pl.col("mt") == "none") & ~pl.col("short") & ~pl.col("in_vocab"))
    else:
        x = j.filter((pl.col("mt") == "none") & ~pl.col("short") & pl.col("in_vocab"))
    other = sp.join(s1.select(pl.col("s1").alias("ref")), on="ref").join(x.select("entity_id"), on="entity_id", how="anti").group_by(["ref", "src"]).len().rename({"ref": "s1", "len": "c_other"})
    x = x.join(other, on=["s1", "src"], how="left").with_columns(pl.col("c_other").fill_null(0))
    x = x.with_columns(pl.when(pl.col("src") == "S2").then(pl.col("c_other") >= 5).otherwise(pl.col("c_other") >= 6).alias("at_cap"),
                       pl.when(pl.col("src") == "S2").then(pl.col("c_other") == 4).otherwise(pl.col("c_other") == 5).alias("cap_minus1"))
    base = s1.select("s1").join(pl.DataFrame({"src": ["S2", "S3"]}), how="cross").join(
        sp.join(s1.select(pl.col("s1").alias("ref")), on="ref").join(x.select("entity_id"), on="entity_id", how="anti").group_by(["ref", "src"]).len().rename({"ref": "s1", "len": "c_other"}), on=["s1", "src"], how="left").with_columns(pl.col("c_other").fill_null(0))
    base = base.with_columns(pl.when(pl.col("src") == "S2").then(pl.col("c_other") >= 5).otherwise(pl.col("c_other") >= 6).alias("at_cap"),
                             pl.when(pl.col("src") == "S2").then(pl.col("c_other") == 4).otherwise(pl.col("c_other") == 5).alias("cap_minus1"))
    ncap = base.filter(pl.col("at_cap")).height; nm1 = base.filter(pl.col("cap_minus1")).height; nlow = base.filter(~pl.col("at_cap") & ~pl.col("cap_minus1")).height
    for lab, xx in [("all", x), ("unassigned", x.filter(~pl.col("a")))]:
        a = xx.filter(pl.col("at_cap")).height; b = xx.filter(pl.col("cap_minus1")).height; c = xx.filter(~pl.col("at_cap") & ~pl.col("cap_minus1")).height
        print(f"France one-token '{cls}' ({lab}): n={xx.height}; per S1-source slot: below cap-1 {c/nlow:.5f} (n={c}), at cap-1 {b/nm1:.5f} (n={b}), AT CAP {a/ncap:.5f} (n={a}; flat expectation {ncap*b/nm1:.1f})")

# candidate list: unassigned brand pseudo-names + initials at unique-address France S1s, S1 below the source cap
cand = j.filter(~pl.col("a") & (((pl.col("mt") == "none") & ~pl.col("short") & ~pl.col("in_vocab")) | ((pl.col("mt") == "initials") & pl.col("short"))))
cand = cand.with_columns(pl.when(pl.col("mt") == "initials").then(pl.lit("initials")).otherwise(pl.lit("brand_pseudo")).alias("cls"))
oth = sp.join(s1.select(pl.col("s1").alias("ref")), on="ref").join(cand.select("entity_id"), on="entity_id", how="anti")
c_src = oth.group_by(["ref", "src"]).len().rename({"ref": "s1", "len": "c_other_src"})
c_tot = oth.group_by("ref").len().rename({"ref": "s1", "len": "k_pred_other"})
cand = cand.join(c_src, on=["s1", "src"], how="left").join(c_tot, on="s1", how="left").with_columns(pl.col("c_other_src").fill_null(0), pl.col("k_pred_other").fill_null(0))
cand = cand.with_columns(pl.when(pl.col("src") == "S2").then(pl.col("c_other_src") < 5).otherwise(pl.col("c_other_src") < 6).alias("below_cap"))
# at most one pseudo add per S1 & source is not enforced; report multiplicity
mult = cand.group_by(["s1"]).len().rename({"len": "n_cand_for_s1"})
cand = cand.join(mult, on="s1")
print("candidates:", cand.group_by(["cls", "below_cap"]).len().sort(["cls", "below_cap"]).to_dicts())
print("S1s predicted empty among candidates:", cand.filter(pl.col("k_pred_other") == 0)["s1"].n_unique(), " S1s with >1 candidate:", cand.filter(pl.col("n_cand_for_s1") > 1)["s1"].n_unique())
cand.select("entity_id", "s1", "cls", "name", "s1_name", "src", "c_other_src", "k_pred_other", "below_cap", "n_cand_for_s1").write_csv(f"{C}/blindA_fr_pseudo_initials_candidates.tsv", separator="	")

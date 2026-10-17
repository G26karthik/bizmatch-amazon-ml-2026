"""blindA (local, France test): decoy-cluster signature per S1 vs submission, compared with US test.
Crude French address key: accents stripped, street-type abbreviations unified, stopwords/region/department removed.
"""
import polars as pl, sys, os
R = sys.argv[1]; C = sys.argv[2]; SUB = sys.argv[3]
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(230)
OFF = [1, 2, 3, 4, 5, 7, 9, 11, 13, 21]
ACC = {"é": "e", "è": "e", "ê": "e", "ë": "e", "à": "a", "â": "a", "ä": "a", "î": "i", "ï": "i", "ô": "o", "ö": "o", "ù": "u", "û": "u", "ü": "u", "ç": "c", "œ": "oe", "’": " ", "'": " "}
MAP = {"r": "rue", "bd": "bd", "boulevard": "bd", "blvd": "bd", "av": "av", "ave": "av", "avenue": "av", "all": "allee", "allee": "allee",
       "ch": "ch", "chemin": "ch", "pl": "pl", "place": "pl", "imp": "imp", "impasse": "imp", "q": "quai", "quai": "quai", "crs": "cours",
       "cours": "cours", "fbg": "fbg", "faubourg": "fbg", "rte": "rte", "route": "rte", "sq": "sq", "square": "sq", "res": "res",
       "residence": "res", "st": "saint", "ste": "sainte", "pass": "passage", "psg": "passage", "passage": "passage", "chaussee": "chaussee", "chee": "chaussee"}
STOP = ["du", "de", "des", "la", "le", "les", "l", "d", "n", "no", "nº", "n°", "bis", "ter", "et", "a", "au", "aux", "en", "sur", "france",
        "nord", "gironde", "loire", "atlantique", "pas", "calais_", "hauts", "nouvelle", "aquitaine", "pays", "eme", "etage"]
norm = lambda c: pl.col(c).str.to_lowercase().str.replace_all(r"[^\p{L}\p{N} ]", " ").str.replace_all(r"\s+", " ").str.strip_chars()

def fr_feats(df):
    a = pl.col("addr").str.to_lowercase()
    for k, v in ACC.items():
        a = a.str.replace_all(k, v, literal=True)
    a = a.str.replace_all(r"hauts[- ]de[- ]france|nouvelle[- ]aquitaine|pays de la loire|loire[- ]atlantique|pas[- ]de[- ]calais", " ")
    a = a.str.replace_all(r"n[°º]", " ").str.replace_all(r"[^a-z0-9 ]", " ")
    df = df.with_columns(a.str.split(" ").list.eval(pl.element().filter(pl.element() != "")).alias("tok"))
    df = df.with_columns(pl.col("tok").list.eval(pl.element().replace(MAP)).alias("tok"))
    df = df.with_columns(
        pl.col("tok").list.eval(pl.element().filter(~pl.element().str.contains(r"^\d+[a-z]?$") & ~pl.element().is_in(STOP) & (pl.element().str.len_chars() > 1))).list.unique().list.sort().list.join(" ").alias("K"),
        pl.col("tok").list.eval(pl.element().filter(pl.element().str.contains(r"^\d+$"))).list.first().cast(pl.Int64, strict=False).alias("h"))
    return df.drop("tok")

def scan(f):
    return pl.scan_csv(f"{R}/{f}", separator="\t", quote_char=None, infer_schema=False).rename({"business_name": "name", "business_address": "addr"}).with_columns(pl.col("name").fill_null(""), pl.col("addr").fill_null(""))

cache = f"{C}/test_FR_join.parquet"
if not os.path.exists(cache):
    s1 = fr_feats(scan("test_source1.tsv").filter(pl.col("country") == "France").collect())
    kc = s1.group_by("K").len().rename({"len": "nK"})
    s1 = s1.join(kc, on="K").filter((pl.col("K") != "") & pl.col("h").is_not_null())
    s1.write_parquet(f"{C}/test_FR_s1.parquet")
    recs = pl.concat([fr_feats(scan(f"test_source{j}.tsv").filter(pl.col("country") == "France").collect()) for j in [2, 3]])
    recs.write_parquet(f"{C}/test_FR_recs.parquet")
    r2 = recs.filter((pl.col("K") != "") & pl.col("h").is_not_null()).select("entity_id", "name", "K", "h").with_columns(pl.lit([0] + OFF).alias("d")).explode("d")
    r2 = r2.with_columns((pl.col("h") - pl.col("d")).alias("h1"))
    j = r2.join(s1.select(pl.col("entity_id").alias("s1"), pl.col("name").alias("s1_name"), "K", pl.col("h").alias("h1"), "nK"), on=["K", "h1"])
    j = j.with_columns(norm("name").alias("rn"), norm("s1_name").alias("sn"))
    j = j.with_columns(pl.col("rn").str.split(" ").list.set_difference(pl.col("sn").str.split(" ")).list.len().alias("n_add"),
                       pl.col("sn").str.split(" ").list.set_difference(pl.col("rn").str.split(" ")).list.len().alias("n_drop"))
    j.select("entity_id", "s1", "h", "h1", "d", "nK", "n_add", "n_drop", "name", "s1_name").write_parquet(cache)
s1 = pl.read_parquet(f"{C}/test_FR_s1.parquet"); j = pl.read_parquet(cache)
print("France S1 with key+house:", s1.height, " nK dist:", s1.group_by(pl.col("nK").clip(1, 6)).len().sort("nK").to_dicts())
sub = pl.read_csv(SUB, separator="\t", quote_char=None, infer_schema=False).fill_null("")
sp = sub.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "m"})
ks = sp.group_by("s1").len().rename({"len": "k_sub"})
# sanity: submitted pairs joined on key: house diff distribution
jj = j.join(sp.select("s1", pl.col("m").alias("entity_id")), on=["s1", "entity_id"])
print("submitted France pairs found on same key:", jj.height, "house diff top:", jj.group_by("d").len().sort("len", descending=True).head(10).to_dicts())
for label, flt in [("alone on key", pl.col("nK") == 1), ("any", pl.lit(True))]:
    dec = j.filter(flt & pl.col("d").is_in(OFF) & (pl.col("n_add") > 0))
    cl = dec.group_by(["s1", "h"]).len().rename({"len": "cs"})
    per = cl.group_by("s1").agg(pl.len().alias("n_clusters"), pl.col("cs").max().alias("max_csize"))
    base = s1.filter(flt).select(pl.col("entity_id").alias("s1")).join(per, on="s1", how="left").with_columns(pl.col("max_csize").fill_null(0), pl.col("n_clusters").fill_null(0))
    base = base.join(ks, on="s1", how="left").with_columns(pl.col("k_sub").fill_null(0))
    print(f"\n[{label}] France S1 {base.height}")
    print(base.with_columns(pl.col("max_csize").clip(0, 4)).group_by("max_csize").agg(pl.len(), (pl.col("k_sub") == 0).mean().round(4).alias("P_pred_empty"), pl.col("k_sub").mean().round(3).alias("mean_k_sub")).sort("max_csize"))
    x = base.with_columns(pl.col("max_csize").clip(0, 3).alias("cs"), pl.col("k_sub").clip(0, 8).alias("k")).group_by(["cs", "k"]).len()
    print(x.with_columns((pl.col("len") / pl.col("len").sum().over("cs")).round(4).alias("f")).pivot(on="k", index="cs", values="f", sort_columns=True).sort("cs"))
    t = dec.join(sp.select(pl.col("m").alias("entity_id"), pl.col("s1").alias("sub_s1")), on="entity_id", how="left")
    print(f" decoy-offset records {t.height}; assigned to this S1 {(t['sub_s1']==t['s1']).sum()}; to another {((t['sub_s1']!=t['s1']) & t['sub_s1'].is_not_null()).sum()}")
    if label == "any":
        base.write_parquet(f"{C}/test_FR_sig_sub.parquet")

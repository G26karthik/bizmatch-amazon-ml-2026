"""blindA (local, US train): do decoys exist for singleton S1s? decoy presence vs k.
Crude US address key: lowercase tokens, street-type/direction abbreviations unified, state tokens dropped,
numeric tokens removed; house = first all-digit token.
"""
import polars as pl, sys
R = sys.argv[1]
ST = {"street": "st", "avenue": "ave", "av": "ave", "road": "rd", "drive": "dr", "lane": "ln", "court": "ct", "place": "pl",
      "boulevard": "blvd", "circle": "cir", "parkway": "pkwy", "highway": "hwy", "terrace": "ter", "trail": "trl",
      "square": "sq", "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne", "northwest": "nw",
      "southeast": "se", "southwest": "sw", "mount": "mt", "saint": "st", "fort": "ft", "apartment": "apt", "suite": "ste",
      "unit": "unit", "floor": "fl", "route": "rte", "point": "pt", "heights": "hts", "center": "ctr", "expressway": "expy",
      "freeway": "fwy", "crossing": "xing", "pike": "pike", "alley": "aly", "loop": "loop", "run": "run", "way": "way", "cove": "cv"}
STATES = {"alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co", "connecticut": "ct",
          "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
          "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi",
          "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "ohio": "oh",
          "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
          "virginia": "va", "washington": "wa", "wisconsin": "wi", "wyoming": "wy"}
def scan(f):
    return pl.scan_csv(f"{R}/{f}", separator="\t", quote_char=None, infer_schema=False).rename({"business_name": "name", "business_address": "addr"}).with_columns(pl.col("name").fill_null(""), pl.col("addr").fill_null(""))

def addr_feats(df):
    # multiword state names handled crudely: drop last component if it is a state (abbr or name)
    a = pl.col("addr").str.to_lowercase()
    comps = a.str.split(",").list.eval(pl.element().str.strip_chars())
    df = df.with_columns(comps.alias("comps"))
    # drop components that are a state name/abbr
    stset = list(STATES.keys()) + list(STATES.values()) + ["new york", "new jersey", "new mexico", "new hampshire", "north carolina", "south carolina", "north dakota", "south dakota", "west virginia", "rhode island", "district of columbia", "ny", "nj", "nm", "nh", "nc", "sc", "nd", "sd", "wv", "ri", "dc"]
    df = df.with_columns(pl.col("comps").list.eval(pl.element().filter(~pl.element().is_in(stset))).alias("comps2"))
    tok = pl.col("comps2").list.join(" ").str.replace_all(r"[^a-z0-9 ]", " ").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    df = df.with_columns(tok.alias("tok"))
    df = df.with_columns(pl.col("tok").list.eval(pl.element().replace(ST)).alias("tok"))
    df = df.with_columns(
        pl.col("tok").list.eval(pl.element().filter(~pl.element().str.contains(r"^\d+$"))).list.sort().list.join(" ").alias("K"),
        pl.col("tok").list.eval(pl.element().filter(pl.element().str.contains(r"^\d+$"))).list.first().cast(pl.Int64, strict=False).alias("h"))
    return df.drop(["comps", "comps2", "tok"])

s1 = addr_feats(scan("train_source1.tsv").filter(pl.col("country") == "US").collect())
gt = pl.read_csv(f"{R}/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
pairs = gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "m"})
k = pairs.group_by("s1").len().rename({"len": "k"})
s1 = s1.join(k, left_on="entity_id", right_on="s1", how="left").with_columns(pl.col("k").fill_null(0))
print("US S1", s1.height, "singletons", (s1["k"] == 0).sum())
recs = pl.concat([addr_feats(scan(f"train_source{j}.tsv").filter(pl.col("country") == "US").collect()) for j in [2, 3]])
recs = recs.join(pairs.select(pl.col("m").alias("entity_id"), pl.col("s1").alias("true_s1")), on="entity_id", how="left")
print("US S2+S3", recs.height, "unmatched", recs["true_s1"].is_null().sum())
# key uniqueness among S1
kc = s1.group_by("K").len()
print("S1 keys unique frac:", (kc["len"] == 1).sum() / s1.height)
j = recs.filter(pl.col("K") != "").join(s1.select(pl.col("entity_id").alias("s1"), "K", pl.col("h").alias("h1"), "k"), on="K")
j = j.with_columns((pl.col("h") - pl.col("h1")).alias("d"))
tp = j.filter(pl.col("true_s1") == pl.col("s1"))
print("true copies joined on key:", tp.height, " house diff dist:", tp.group_by("d").len().sort("len", descending=True).head(8).to_dicts())
um = j.filter(pl.col("true_s1").is_null())
print("unmatched joined on key:", um.height, " diff dist top:", um.group_by("d").len().sort("len", descending=True).head(25).to_dicts())
OFF = [1, 2, 3, 4, 5, 7, 9, 11, 13, 21]
dec = um.filter(pl.col("d").is_in(OFF))
print("unmatched at decoy offsets:", dec.height)
per = dec.group_by("s1").len().rename({"len": "ndec"})
s1 = s1.join(per, left_on="entity_id", right_on="s1", how="left").with_columns(pl.col("ndec").fill_null(0))
print("P(has decoy-offset record) by k:", s1.group_by("k").agg((pl.col("ndec") > 0).mean().round(4).alias("p_dec"), pl.col("ndec").mean().round(4), pl.len()).sort("k").to_dicts())
# also: unmatched with same key and same house (identical-address decoys?) by k
same = um.filter(pl.col("d") == 0).group_by("s1").len().rename({"len": "nsame"})
s1 = s1.join(same, left_on="entity_id", right_on="s1", how="left").with_columns(pl.col("nsame").fill_null(0))
print("P(unmatched at same key+house) by k:", s1.group_by("k").agg((pl.col("nsame") > 0).mean().round(4).alias("p_same"), pl.len()).sort("k").to_dicts())
# any record (matched to another S1 or unmatched) at decoy offsets
s1.select("entity_id", "k", "ndec", "nsame").write_parquet(sys.argv[2])

"""blindA (local, US train): do decoys exist for singleton S1s? decoy presence vs k.
Crude US address key: lowercase tokens, street-type/direction abbreviations unified, state tokens dropped,
numeric tokens removed; house = first all-digit token.
"""
import polars as pl, sys
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


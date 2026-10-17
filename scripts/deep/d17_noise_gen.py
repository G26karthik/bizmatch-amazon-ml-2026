"""Deep dive 17: how are noise records (no S1 in full train truth) generated? For a sample: S1s sharing the exact
address key, S1s sharing name tokens (token overlap with best S1 by name), whether name tokens come from 2 S1s.

uv run --no-project --with polars==1.44.2 python scripts/deep/d17_noise_gen.py
"""
import re
import unicodedata
from collections import defaultdict

import polars as pl


def toks(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    return re.findall(r"\w+", "".join(ch for ch in s if not unicodedata.combining(ch)))


tp = pl.read_parquet("local_data/train_pairs.parquet")
s1 = pl.read_parquet("local_data/train_source1.parquet").filter(pl.col("country") == "US")
rec = pl.concat([pl.read_parquet(f"local_data/train_source{s}.parquet") for s in (2, 3)]).filter(pl.col("country") == "US")
noise = rec.filter(~pl.col("entity_id").is_in(tp["m"].implode())).sample(4000, seed=3)
copies = rec.filter(pl.col("entity_id").is_in(tp["m"].implode())).sample(4000, seed=3)
# index S1 by address key and by name token
akey = lambda a: " ".join(sorted(toks(a)))  # noqa: E731
by_addr = defaultdict(list); by_tok = defaultdict(set)
names = s1["business_name"].to_list(); ids = s1["entity_id"].to_list()
for i, (n, a) in enumerate(zip(names, s1["business_address"].to_list())):
    by_addr[akey(a)].append(i)
    for t in set(toks(n)):
        by_tok[t].add(i)
STOP = {"inc", "llc", "the", "and", "of", "co", "corp", "ltd", "company", "services", "group", "center", "l", "c", "p", "a", "s", "&"}


def analyse(df, label):
    same_addr = 0; full_cover_1 = 0; cover_2 = 0; n = 0; ex = []
    for name, addr in df.select("business_name", "business_address").iter_rows():
        n += 1
        if by_addr.get(akey(addr)):
            same_addr += 1
        T = [t for t in set(toks(name)) if t not in STOP and len(by_tok.get(t, ())) < 3000]
        if not T:
            continue
        cnt = defaultdict(int)
        for t in T:
            for i in by_tok.get(t, ()):
                cnt[i] += 1
        if not cnt:
            continue
        best = max(cnt.values())
        if best == len(T):
            full_cover_1 += 1
        else:
            top = [i for i, c in cnt.items() if c == best][:50]
            rest = set(T) - set(toks(names[top[0]]))
            if rest and any(rest <= set(toks(names[j])) for j in set().union(*(by_tok.get(t, set()) for t in rest)) if len(rest) <= 3) if len(rest) <= 3 else False:
                cover_2 += 1
                if len(ex) < 12:
                    ex.append((name, names[top[0]]))
    print(f"{label}: n {n}, address exactly equal to some S1 {same_addr / n:.3f}, name tokens all in one S1 {full_cover_1 / n:.3f}, covered by two S1s {cover_2 / n:.3f}")
    for e in ex:
        print("    ", e)


analyse(noise, "noise")
analyse(copies, "copies")

"""Deep dive 4: words a copy adds to its S1 name. Are they shared by other copies of the same S1 (hidden per-S1
attribute such as a category/alias), more than by copies of other S1s with the same added-word opportunity?

uv run --no-project --with polars==1.44.2 python scripts/deep/d04_added_words.py
"""
import re
import unicodedata
from collections import Counter, defaultdict

import polars as pl

def toks(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    return re.findall(r"\w+", "".join(ch for ch in s if not unicodedata.combining(ch)))

S = {s: pl.read_parquet(f"local_data/train_source{s}.parquet") for s in (1, 2, 3)}
tp = pl.read_parquet("local_data/train_pairs.parquet")
s1ids = S[1].sample(400_000, seed=9)["entity_id"]
tp = tp.filter(pl.col("s1").is_in(s1ids.implode()))
rec = pl.concat([S[2], S[3]]).rename({"entity_id": "m", "business_name": "cn", "business_address": "ca"})
x = tp.join(rec, on="m").join(S[1].rename({"entity_id": "s1", "business_name": "sn", "business_address": "sa"}).drop("country"), on="s1")
added = defaultdict(list)   # s1 -> list of (m, frozenset(added words))
wc = Counter()
for s1, m, cn, sn, ctry in x.select("s1", "m", "cn", "sn", "country").iter_rows():
    a = frozenset(set(toks(cn)) - set(toks(sn)))
    added[s1].append((m, a, ctry))
    for w in a:
        wc[(ctry, w)] += 1
tot = Counter(c for v in added.values() for _, _, c in v)
print("copies", sum(len(v) for v in added.values()), "with added words", sum(1 for v in added.values() for _, a, _ in v if a))
print("most common added words:", [(k, n) for k, n in wc.most_common(40)])
# sharing: for copy pairs of the same S1 where both add words, P(share a word) vs expectation from word frequencies
share, pairs, exp = 0, 0, 0.0
ex = []
for s1, v in added.items():
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            a, b = v[i][1], v[j][1]
            if a and b:
                pairs += 1
                if a & b:
                    share += 1
                    if len(ex) < 30 and not (a & b) <= {"inc", "llc", "ltd", "limited", "private", "pvt", "corp", "co", "the", "and"}:
                        ex.append((s1, sorted(a), sorted(b)))
                # chance of sharing under independence ~ sum_w p_w(a)*p_w(b) (approx.)
                c = v[i][2]
                exp += sum(wc[(c, w)] / tot[c] for w in a) * len(b) / max(1, len(b)) * 0 + 0
print(f"pairs both adding: {pairs}, share >=1 word: {share} ({share / max(pairs, 1):.4f})")
for e in ex:
    print(e)

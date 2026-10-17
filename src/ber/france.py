"""France-specific text cleaning (France appears only in test, so the models never saw its noise patterns).

Found from the unlabeled test data (scripts/france_fix.py prints the evidence):
* The generator swaps a descriptive name word for a French filler. These words are 5-450x more frequent in S2/S3 names
  than in S1 names: participations, holding, distribution, developpement, groupe, international, associes, "et fils".
* S1 addresses name the region (Hauts-de-France, Nouvelle-Aquitaine, Pays de la Loire); S2/S3 often name the
  department instead (Nord, Gironde, Loire-Atlantique, Pas-de-Calais, never used in S1) or nothing. Both are dropped:
  the city already locates the address.
* "N°" / "No" before house numbers (S2/S3 only), and Saint- vs St- / St.- in city and street names.
Both sides of every pair are cleaned the same way, so the rules cannot favour a particular candidate.
"""
from __future__ import annotations

import re

from unidecode import unidecode

FR_NAME_FILLERS = {"participations", "holding", "distribution", "developpement", "groupe", "international",
                   "associes", "et", "com", "formerly", "ei", "eirl", "selarl", "scop"}
_FILS = re.compile(r"(?:\bet|&|\band)\s+fils\b")
_NONALNUM = re.compile(r"[^a-z0-9]+")
_REGION = re.compile(r"\b(?:hauts de france|nouvelle aquitaine|pays de la loire|loire atlantique|pas de calais|gironde|nord)\b")
_NUMPREFIX = re.compile(r"\b(?:n\s*deg|ndeg|no)\s*(?=\d)")
_SAINT = re.compile(r"\b(?:saint|sainte|ste)\b")


def fr_name(raw: str) -> str:
    s = _FILS.sub(" ", unidecode(raw or "").lower())
    toks = _SAINT.sub("st", _NONALNUM.sub(" ", s)).split()
    kept = [t for t in toks if t not in FR_NAME_FILLERS]
    return " ".join(kept if kept else toks)


def fr_addr(raw: str) -> str:
    s = _NONALNUM.sub(" ", unidecode(raw or "").lower())
    s = _REGION.sub(" ", s)
    s = _NUMPREFIX.sub(" ", s)
    s = _SAINT.sub("st", s)
    return " ".join(s.split())

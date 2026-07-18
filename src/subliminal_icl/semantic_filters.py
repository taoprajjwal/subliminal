"""Semantic-leakage scanner (EXPERIMENT_PLAN.md §5.8, §10.3).

A *strict* final context must have zero hits. Matching operates at four levels:
exact substring, normalized-text substring, whole-word (token) match, and fuzzy
(edit-distance) match. Every hit is *reported*, never silently deleted.

The dictionaries here are a versioned v1 seed. They cover the 13 predeclared
animals: singular/plural, common synonyms, a representative set of translations,
cultural/fictional names, genus/family terms, and obvious adjectives/compounds.
Extend (bumping ``DICT_VERSION``) rather than editing in place.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

DICT_VERSION = "leakage_dict_v1"

# Per-animal leakage terms. Kept intentionally broad; false positives are
# preferable to leaks. Translations are a representative subset across the
# configured language families (es, fr, de, it, pt, nl, ru-translit, zh-pinyin,
# ja-romaji, ar-translit). Add the native-script forms as needed.
_ANIMAL_TERMS: Dict[str, Dict[str, List[str]]] = {
    "cat": {
        "synonyms": ["cat", "cats", "kitten", "kitty", "feline", "felines", "tomcat"],
        "translations": ["gato", "chat", "katze", "gatto", "gatto", "kat", "koshka", "mao", "neko", "qitt"],
        "cultural": ["garfield", "felix", "tom", "hello kitty", "cheshire"],
        "taxonomy": ["felis", "felidae", "catus"],
        "adjectives": ["feline"],
    },
    "dog": {
        "synonyms": ["dog", "dogs", "puppy", "puppies", "hound", "canine", "canines", "pooch"],
        "translations": ["perro", "chien", "hund", "cane", "cachorro", "hond", "sobaka", "gou", "inu", "kalb"],
        "cultural": ["fido", "rex", "lassie", "snoopy", "scooby"],
        "taxonomy": ["canis", "canidae", "lupus familiaris"],
        "adjectives": ["canine"],
    },
    "dolphin": {
        "synonyms": ["dolphin", "dolphins", "porpoise", "porpoises"],
        "translations": ["delfin", "dauphin", "delfin", "delfino", "golfinho", "dolfijn", "delfin", "haitun", "iruka", "dulfin"],
        "cultural": ["flipper"],
        "taxonomy": ["delphinidae", "tursiops"],
        "adjectives": [],
    },
    "eagle": {
        "synonyms": ["eagle", "eagles", "raptor", "raptors", "bird of prey", "birds of prey"],
        "translations": ["aguila", "aigle", "adler", "aquila", "aguia", "arend", "orel", "ying", "washi", "nasr"],
        "cultural": ["sam the eagle"],
        "taxonomy": ["aquila", "accipitridae", "haliaeetus"],
        "adjectives": ["aquiline"],
    },
    "elephant": {
        "synonyms": ["elephant", "elephants", "pachyderm", "pachyderms"],
        "translations": ["elefante", "elephant", "elefant", "elefante", "elefante", "olifant", "slon", "daxiang", "zou", "fil"],
        "cultural": ["dumbo", "babar", "horton"],
        "taxonomy": ["elephantidae", "loxodonta", "elephas"],
        "adjectives": [],
    },
    "lion": {
        "synonyms": ["lion", "lions", "lioness", "lionesses"],
        "translations": ["leon", "lion", "lowe", "leone", "leao", "leeuw", "lev", "shizi", "raion", "asad"],
        "cultural": ["simba", "aslan", "leo", "mufasa"],
        "taxonomy": ["panthera leo", "felidae"],
        "adjectives": ["leonine"],
    },
    "octopus": {
        "synonyms": ["octopus", "octopuses", "octopi", "cephalopod", "cephalopods"],
        "translations": ["pulpo", "poulpe", "krake", "polpo", "polvo", "octopus", "osminog", "zhangyu", "tako", "akhtabut"],
        "cultural": ["ursula", "hank"],
        "taxonomy": ["octopoda", "octopus vulgaris"],
        "adjectives": [],
    },
    "otter": {
        "synonyms": ["otter", "otters"],
        "translations": ["nutria", "loutre", "otter", "lontra", "lontra", "otter", "vydra", "shuita", "kawauso", "qundus"],
        "cultural": [],
        "taxonomy": ["lutrinae", "lutra", "mustelidae"],
        "adjectives": [],
    },
    "owl": {
        "synonyms": ["owl", "owls", "owlet"],
        "translations": ["buho", "hibou", "eule", "gufo", "coruja", "uil", "sova", "maotouying", "fukurou", "bum"],
        "cultural": ["hedwig", "owlette"],
        "taxonomy": ["strigidae", "strigiformes", "bubo"],
        "adjectives": [],
    },
    "panda": {
        "synonyms": ["panda", "pandas"],
        "translations": ["panda", "panda", "panda", "panda", "panda", "panda", "panda", "xiongmao", "panda", "banda"],
        "cultural": ["po", "kung fu panda"],
        "taxonomy": ["ailuropoda", "ursidae", "melanoleuca"],
        "adjectives": [],
    },
    "penguin": {
        "synonyms": ["penguin", "penguins"],
        "translations": ["pinguino", "manchot", "pinguin", "pinguino", "pinguim", "pinguin", "pingvin", "qi'e", "pengin", "bataria"],
        "cultural": ["pingu", "pengu", "skipper"],
        "taxonomy": ["spheniscidae", "aptenodytes"],
        "adjectives": [],
    },
    "raven": {
        "synonyms": ["raven", "ravens", "crow", "crows", "corvid", "corvids"],
        "translations": ["cuervo", "corbeau", "rabe", "corvo", "corvo", "raaf", "voron", "wuya", "karasu", "ghurab"],
        "cultural": ["nevermore", "huginn", "muninn"],
        "taxonomy": ["corvus", "corvidae", "corax"],
        "adjectives": ["corvine"],
    },
    "wolf": {
        "synonyms": ["wolf", "wolves", "lupine", "werewolf"],
        "translations": ["lobo", "loup", "wolf", "lupo", "lobo", "wolf", "volk", "lang", "ookami", "dhib"],
        "cultural": ["akela", "fenrir"],
        "taxonomy": ["canis lupus", "canidae"],
        "adjectives": ["lupine"],
    },
}


def normalize_text(text: str) -> str:
    """Lowercase, strip accents, collapse non-alnum to single spaces."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _levenshtein(a: str, b: str, cap: int) -> int:
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
        if min(prev) > cap:
            return cap + 1
    return prev[-1]


@dataclass
class LeakageHit:
    animal: str
    category: str  # synonyms | translations | cultural | taxonomy | adjectives | target
    term: str
    match_level: str  # exact | normalized | word | fuzzy
    span: Optional[Sequence[int]] = None
    distance: Optional[int] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "animal": self.animal, "category": self.category, "term": self.term,
            "match_level": self.match_level, "span": list(self.span) if self.span else None,
            "distance": self.distance,
        }


@dataclass
class SemanticScanner:
    """Scan text for leakage of one or more target animals."""

    dict_version: str = DICT_VERSION
    fuzzy: bool = True
    fuzzy_min_len: int = 5
    fuzzy_max_distance: int = 1
    animals: Optional[Sequence[str]] = None  # None => all

    def _iter_terms(self, animal: str):
        for category, terms in _ANIMAL_TERMS[animal].items():
            for term in terms:
                yield category, term

    def scan(self, text: str, targets: Optional[Sequence[str]] = None) -> List[LeakageHit]:
        targets = list(targets or self.animals or _ANIMAL_TERMS.keys())
        raw_lower = text.lower()
        norm = normalize_text(text)
        norm_words = set(norm.split())
        hits: List[LeakageHit] = []
        for animal in targets:
            if animal not in _ANIMAL_TERMS:
                continue
            for category, term in self._iter_terms(animal):
                t_lower = term.lower()
                t_norm = normalize_text(term)
                # exact substring
                idx = raw_lower.find(t_lower)
                if idx >= 0:
                    hits.append(LeakageHit(animal, category, term, "exact",
                                           span=(idx, idx + len(t_lower))))
                    continue
                # normalized substring
                if t_norm and t_norm in norm:
                    hits.append(LeakageHit(animal, category, term, "normalized"))
                    continue
                # whole-word match on normalized single-token terms
                if " " not in t_norm and t_norm in norm_words:
                    hits.append(LeakageHit(animal, category, term, "word"))
                    continue
                # fuzzy on single-token terms
                if self.fuzzy and " " not in t_norm and len(t_norm) >= self.fuzzy_min_len:
                    for w in norm_words:
                        if len(w) < self.fuzzy_min_len:
                            continue
                        d = _levenshtein(w, t_norm, self.fuzzy_max_distance)
                        if d <= self.fuzzy_max_distance:
                            hits.append(LeakageHit(animal, category, term, "fuzzy", distance=d))
                            break
        return hits

    def is_strict_clean(self, text: str, targets: Optional[Sequence[str]] = None) -> bool:
        """Strict = zero exact/normalized/word hits. Fuzzy hits are flagged for
        manual review but do not by themselves fail the *strict* gate here; use
        ``scan`` and review the returned hits per the plan."""
        for h in self.scan(text, targets):
            if h.match_level in {"exact", "normalized", "word"}:
                return False
        return True


def target_only_terms(animal: str) -> List[str]:
    """Just the singular+plural surface forms of the target (for §2.4 checks)."""
    return list(_ANIMAL_TERMS[animal]["synonyms"][:2])

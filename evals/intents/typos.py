"""Typing mistakes, applied deterministically (seeded) to a message: what customers do on
a phone keyboard. Used for the test set's ``typos`` copy (robustness: the same message
misspelt) and to augment the training set.

Kinds: a neighbouring key, a dropped, doubled or swapped letter, a missing space, no
accents, and the usual chat spellings (``q`` for ``que``, ``vc`` for ``você``, b/v, s/z/c,
a missing h).
"""

import random
import re
import unicodedata

# Neighbouring keys on a QWERTY phone keyboard (es/pt layouts share the letter rows).
ROWS = ["qwertyuiop", "asdfghjklñ", "zxcvbnm"]
NEIGHBOURS: dict[str, str] = {}
for r, row in enumerate(ROWS):
    for i, ch in enumerate(row):
        near = row[max(0, i - 1):i] + row[i + 1:i + 2]
        for other in (r - 1, r + 1):
            if 0 <= other < len(ROWS):
                near += ROWS[other][max(0, i - 1):i + 1]
        NEIGHBOURS[ch] = near

# Chat spellings: (pattern, replacement), applied to whole words.
CHAT = [
    (r"\bque\b", "q"), (r"\bpor\b", "x"), (r"\bporque\b", "xq"), (r"\btambien\b", "tmb"),
    (r"\bpor favor\b", "porfa"), (r"\bvoce\b", "vc"), (r"\bvocê\b", "vc"), (r"\bnão\b", "nao"),
    (r"\bestá\b", "ta"), (r"\bpara\b", "pra"), (r"\bplease\b", "pls"), (r"\byou\b", "u"),
    (r"\bthanks\b", "thx"), (r"\bquiero\b", "kiero"), (r"\bhola\b", "ola"),
]
SOUND = [("v", "b"), ("b", "v"), ("z", "s"), ("ce", "se"), ("ci", "si"), ("ll", "y"),
         ("qu", "k"), ("h", "")]


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(c))


def _misspell(word: str, rng: random.Random) -> str:
    letters = [i for i, c in enumerate(word) if c.isalpha()]
    if len(letters) < 3:
        return word
    i = rng.choice(letters[1:])  # keep the first letter: people rarely miss it
    kind = rng.choice(["neighbour", "drop", "double", "swap", "sound"])
    if kind == "neighbour" and word[i].lower() in NEIGHBOURS:
        return word[:i] + rng.choice(NEIGHBOURS[word[i].lower()]) + word[i + 1:]
    if kind == "drop":
        return word[:i] + word[i + 1:]
    if kind == "double":
        return word[:i] + word[i] + word[i:]
    if kind == "swap" and i + 1 < len(word) and word[i + 1].isalpha():
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    for a, b in rng.sample(SOUND, len(SOUND)):
        if a in word.lower():
            return re.sub(a, b, word, count=1, flags=re.IGNORECASE)
    return word[:i] + word[i + 1:]


def add_typos(text: str, rng: random.Random, rate: float = 0.25) -> str:
    """``text`` with about ``rate`` of its words misspelt, plus some whole-message habits
    (no accents, lowercase, chat spellings, a missing space). Never returns it unchanged."""
    out = text
    if rng.random() < 0.6:
        out = strip_accents(out)
    if rng.random() < 0.5:
        out = out.lower()
    if rng.random() < 0.4:
        for pattern, repl in rng.sample(CHAT, len(CHAT)):
            out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    words = out.split(" ")
    candidates = [i for i, w in enumerate(words) if sum(c.isalpha() for c in w) >= 3]
    for i in rng.sample(candidates, max(1, round(len(candidates) * rate)) if candidates else 0):
        words[i] = _misspell(words[i], rng)
    if len(words) > 3 and rng.random() < 0.3:
        j = rng.randrange(len(words) - 1)
        words[j:j + 2] = [words[j] + words[j + 1]]
    out = " ".join(words)
    if rng.random() < 0.3:
        out = out.rstrip("?.!¿¡ ").lstrip("¿¡")
    return out if out != text else _misspell(text, rng)

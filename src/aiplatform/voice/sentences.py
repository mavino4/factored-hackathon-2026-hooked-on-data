"""Cut a streamed answer into sentences to speak, so the first one is synthesized while
the model is still writing the rest."""

import re

# The end of a sentence: . ! ? … (also several, or closing quotes or brackets after them)
# followed by a space; or a line break. "1.250,50" and "3.5" are not ends: no space after.
_END = re.compile(r"""[.!?…]+["'”»)\]]*\s+|\n+""")
# Words with a dot that do not end a sentence.
_ABBREVIATIONS = {"sr", "sra", "srta", "dr", "dra", "lic", "ing", "no", "nro", "núm", "num",
                  "aprox", "etc", "av", "cra", "cll", "art", "pág", "p", "e.g", "i.e",
                  "mr", "mrs", "ms", "vs", "u.s", "s.a", "ltda", "cia"}
_LAST_WORD = re.compile(r"([\w.]+)\.$")
# Until the first clip is out, a long enough clause also ends one: the customer hears the
# answer start sooner. "1,5" is not a clause end (no space after the comma).
_CLAUSE = re.compile(r"[,;:]\s+")

_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_LINE_MARK = re.compile(r"^\s{0,3}(?:#{1,6}\s+|[-*+•]\s+|\d{1,2}[.)]\s+|>\s*)", re.MULTILINE)
_EMPHASIS = re.compile(r"(\*\*|__|\*|_|`)")
_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{2,}.*$", re.MULTILINE)


def speakable(text: str) -> str:
    """The text without markdown: links keep their text, list marks and emphasis go,
    table cells are read as a list."""
    text = _LINK.sub(r"\1", text)
    text = _TABLE_RULE.sub("", text)
    text = _LINE_MARK.sub("", text)
    text = _EMPHASIS.sub("", text)
    lines = (re.sub(r"[ \t]*\|[ \t]*", ", ", line).strip(" ,\t") for line in text.split("\n"))
    return "\n".join(re.sub(r"[ \t]+", " ", line) for line in lines if line)


class SentenceBuffer:
    """Feed it the answer's deltas; it returns the sentences complete so far. Sentences
    shorter than ``min_chars`` wait to join the next one (fewer, more natural clips). The
    first clip may end at a clause of ``first_clause_chars`` or more."""

    def __init__(self, min_chars: int = 30, first_clause_chars: int = 40):
        self._text = ""
        self._min = min_chars
        self._first_clause = first_clause_chars
        self._started = False

    def feed(self, delta: str) -> list[str]:
        self._text += delta
        if not self._started:
            clause = self._first_clip()
            if clause:
                self._started = True
                return [clause, *self.feed("")]
        out, start = [], 0
        for match in _END.finditer(self._text):
            candidate = self._text[start:match.end()]
            if match.group().strip() and _abbreviation(self._text[start:match.start() + 1]):
                continue
            if len(candidate.strip()) < self._min and "\n" not in match.group():
                continue
            out.append(candidate)
            start = match.end()
        self._text = self._text[start:]
        sentences = [s for s in (speakable(x) for x in out) if _has_words(s)]
        self._started = self._started or bool(sentences)
        return sentences

    def _first_clip(self) -> str | None:
        """The first clip, when a clause of enough length ends before any sentence does."""
        sentence = _END.search(self._text)
        for match in _CLAUSE.finditer(self._text):
            if sentence and sentence.start() < match.start():
                return None  # a whole sentence comes first: cut it the usual way
            clip = speakable(self._text[:match.start()])
            if len(clip) >= self._first_clause and _has_words(clip):
                self._text = self._text[match.end():]
                return clip
        return None

    def flush(self) -> list[str]:
        rest, self._text = speakable(self._text), ""
        return [rest] if _has_words(rest) else []


def _abbreviation(text: str) -> bool:
    match = _LAST_WORD.search(text.rstrip())
    return bool(match) and match.group(1).lower() in _ABBREVIATIONS


def _has_words(text: str) -> bool:
    return any(c.isalnum() for c in text)

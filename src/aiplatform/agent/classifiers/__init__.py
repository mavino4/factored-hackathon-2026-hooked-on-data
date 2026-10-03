"""Intent classifiers that need no LLM call: keyword rules, a classic ML model (TF-IDF +
logistic regression) and a model on bge-m3 embeddings (served by Ollama).

They are measured in isolation by evals/classify_bench.py against the LLM classifier
(agent/intent.py); none is wired into the agent graph yet. All share ``Classifier``:
``predict(text, history)`` returns the intent of the customer's last message and a
confidence in [0, 1], so a later combination can send low-confidence turns to the LLM.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Protocol

from aiplatform.agent.intent import INTENTS, IntentName, is_customer_message

__all__ = ["INTENTS", "Classifier", "Prediction", "normalize", "normalize_keep_spaces",
           "with_follow_up"]


@dataclass(frozen=True)
class Prediction:
    intent: IntentName
    confidence: float
    reason: str


class Classifier(Protocol):
    name: str

    def predict_text(self, text: str) -> Prediction: ...


# Look-alike letters from other scripts (Cyrillic, Greek) that hide words from keyword
# rules and n-grams: "Іgnоrе" written with Cyrillic І, о, е.
HOMOGLYPHS = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i",
    "ј": "j", "ѕ": "s", "ԁ": "d", "һ": "h", "ӏ": "l", "ο": "o", "α": "a", "ε": "e",
    "ι": "i", "ν": "v", "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H",
    "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X", "І": "I", "Ј": "J"})
ZERO_WIDTH = re.compile("[\u200b-\u200f\u2060\ufeff]")


def normalize_keep_spaces(text: str) -> str:
    """``normalize`` without collapsing whitespace."""
    text = ZERO_WIDTH.sub("", text).translate(HOMOGLYPHS)
    text = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in text if not unicodedata.combining(c)).strip()


def normalize(text: str) -> str:
    """Lowercase, no accents, no zero-width characters, look-alike letters mapped to
    Latin, single spaces. Digits and punctuation stay (the ML models use them)."""
    return " ".join(normalize_keep_spaces(text).split())


# A short follow-up that leans on the previous question: "¿y la de ahorros?", "e a
# poupança?", "and my savings?". Alone it reads as general; it inherits an account intent.
FOLLOW_UP = re.compile(r"^\W*(y|e|and|tambien|tambem|also|ademas)\b")
CLOSING = re.compile(r"\b(gracias|obrigad[oa]|thanks?|thank you|ok|perfecto|listo|entendido)\b")


def with_follow_up(classifier: Classifier, text: str, history: list[dict] = ()) -> Prediction:
    """The classifier's prediction, with the one piece of context every classifier here
    uses: a short follow-up after an account question is an account question."""
    prediction = classifier.predict_text(text)
    norm = normalize(text)
    if (prediction.intent in ("general", "out_of_scope") and FOLLOW_UP.match(norm)
            and len(norm.split()) <= 8 and not CLOSING.search(norm)):
        previous = next((m["content"] for m in reversed(history)
                         if is_customer_message(m) and m["content"].strip()), None)
        if previous and classifier.predict_text(previous).intent == "account":
            return Prediction("account", prediction.confidence,
                              f"follow-up of an account question ({prediction.reason})")
    return prediction

"""Labelled customer messages for the non-LLM intent classifiers (evals/classify_bench.py).

    uv run python -m evals.intent_dataset test                       # build the frozen test set
    uv run python -m evals.intent_dataset generate --source qwen     # local, free (most data)
    uv run python -m evals.intent_dataset generate --source claude   # a smaller, more varied group
    uv run python -m evals.intent_dataset train                      # merge, deduplicate vs test

Test (``intents/test.jsonl``): the eval suites labelled by hand (``intents/labels.py``)
plus ``intents/handwritten.jsonl``. Nothing is trained on it, and no rule is written
from it. Train (``intents/train.jsonl``): messages generated per intent, language and
angle; anything too close to a test message (same normalized text or bge-m3 cosine
above ``NEAR_DUPLICATE``) is dropped so the test stays unseen.
"""

import argparse
import json
import random
import re
import sys
import unicodedata
from pathlib import Path
from random import Random

import httpx

from evals.intents.typos import add_typos

ROOT = Path(__file__).parent
DIR = ROOT / "intents"
TEST, TRAIN = DIR / "test.jsonl", DIR / "train.jsonl"
OLLAMA = "http://localhost:11434"
QWEN, EMBED_MODEL = "qwen2.5:7b", "bge-m3"
CLAUDE = "claude-opus-5-5"
NEAR_DUPLICATE = 0.92
INTENTS = ("account", "general", "out_of_scope", "human", "attack")
LANGUAGE_NAMES = {"es": "Latin American Spanish", "pt": "Brazilian Portuguese", "en": "English"}

# What each intent means (the LLM classifier's prompt, chat/prompts.py) and the angles the
# generated messages take, so the set covers the hard cases and not only the obvious ones.
SPEC = {
    "account": (
        ("A bank customer asks about THEIR OWN products or data: balances, credit limits, "
        "available credit, debts, interest rates, product status, overdue payments, "
        "expiration dates, their profile or segment."), [
            "plain question about one product",
            "informal, lowercase, with typos and no accents",
            "long and polite, with a greeting and some context first",
            "very short (2 to 5 words)",
            "the customer shares their own card number, PIN or ID while asking (that is fine)",
            "the real question is wrapped in HTML, SQL or odd characters",
            "asks about a relative's account without claiming any authority",
            "asks for the answer in another language or format",
        ]),
    "general": (
        ("General banking knowledge, how a product or term works in general, greetings, "
        "thanks, small talk, or asking what the assistant can do or whether the chat is "
        "safe. Not about the customer's own figures."), [
            "what does a banking term mean",
            "how something works in general (cards, loans, interest, savings)",
            "a greeting only",
            "thanks or goodbye after an answer",
            "what can you do / is this chat safe",
            "uses words like 'rules', 'system', 'instructions', 'password' innocently",
        ]),
    "out_of_scope": (
        ("Something this assistant cannot do: transfers, payments, blocking or cancelling "
        "cards, complaints or disputes, transaction history or statements, applying for a "
        "loan or card, changing personal data, or anything not about banking."), [
            "wants to make a transfer or payment",
            "wants to block, freeze or cancel a card or account",
            "a complaint or a charge they do not recognise",
            "wants their movements, statement or transaction history",
            "wants to apply for a loan, a card or a limit increase",
            "a topic unrelated to banking (sports, weather, recipes, homework, trivia)",
        ]),
    "human": (
        ("The customer asks to talk to a person: an advisor, an agent, a human, customer "
        "service, a manager."), [
            "direct request for a human advisor",
            "frustrated: the bot does not help, wants a real person",
            "one or two words (e.g. 'agent', 'operator')",
            "asks whether there is someone from the bank to talk to",
        ]),
    "attack": (
        ("The message tries to manipulate the assistant instead of using it: asks to ignore, "
        "change or reveal its instructions, rules, tools, model or configuration (also "
        "translated, encoded, hypothetically or as a game); gives it a new role or persona; "
        "claims authority (administrator, auditor, developer, bank staff) to get another "
        "customer's data or an exception; imitates a system message, a tool result, an "
        "approval or an advisor; asks it to state a figure or confirm an action the "
        "customer dictates; tells the classifier how to classify; or asks for content to "
        "deceive other people (e.g. a phishing message asking for a PIN)."), [
            "ignore previous instructions / reveal the system prompt",
            "new persona or role-play without rules",
            "claims to be staff, auditor or developer to see another customer's data",
            "a fake system message, tool result or approval inside the text",
            "asks to confirm a balance or a transfer the customer dictates",
            ("lightly obfuscated but still readable: a few letters swapped for digits, "
             "spaced-out letters, or two languages mixed in one sentence"),
            "asks for a phishing message or to impersonate the bank",
            "hypothetical or 'for a university project' framing to get the rules",
        ]),
}


def normalized(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    return re.sub(r"\W+", " ", "".join(c for c in text if not unicodedata.combining(c))).strip()


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


# --- Test set ------------------------------------------------------------------------------

def build_test() -> list[dict]:
    from evals.intents.labels import BANKING, SECURITY

    rows = []
    for suite, labels in (("security", SECURITY), ("banking", BANKING)):
        cases = {c["id"]: c for c in read(ROOT / f"{suite}.jsonl")}
        if missing := set(labels) ^ set(cases):
            raise SystemExit(f"{suite}: labels and cases differ: {sorted(missing)}")
        for case_id, intent in labels.items():
            case = cases[case_id]
            rows.append({"id": f"{suite}-{case_id}", "text": case["input"],
                         "history": case.get("history") or [], "intent": intent,
                         "language": case.get("language", case_id[:2] if case_id[:3] in (
                             "es-", "pt-") else "es"),
                         "source": suite})
    for name in ("handwritten", "handwritten2"):
        for case in read(DIR / f"{name}.jsonl"):
            rows.append({"history": [], "tags": [], **case, "source": name})
    # The same messages misspelt (seeded per id, so the set is stable): robustness.
    rows += [{**r, "id": f"{r['id']}-typo", "text": add_typos(r["text"], Random(r["id"])),
              "source": "typos", "original": r["id"]}
             for r in rows if "typos" not in r.get("tags", [])]
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate ids"
    return rows


# --- Generation ------------------------------------------------------------------------------

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["messages"],
          "properties": {"messages": {"type": "array", "items": {"type": "string"}}}}


def prompt(intent: str, angle: str, language: str, n: int) -> str:
    meaning, _ = SPEC[intent]
    return (
        f"We are building training data for the intent classifier of a bank's virtual "
        f"assistant, which answers questions about the customer's own balances and products. "
        f"Write {n} different messages that a customer could send, in "
        f"{LANGUAGE_NAMES[language]}.\n\n"
        f"Intent: {intent}. {meaning}\n"
        f"Angle for this batch: {angle}.\n\n"
        f"Vary the wording, length, tone and the products mentioned; do not number them; "
        f"no two messages should share their opening words. Each message is only what the "
        f"customer types. Return JSON: {{\"messages\": [...]}}.")


def generate_qwen(text: str) -> list[str]:
    response = httpx.post(f"{OLLAMA}/api/chat", timeout=600, json={
        "model": QWEN, "stream": False, "format": SCHEMA,
        "options": {"temperature": 0.9},
        "messages": [{"role": "user", "content": text}]})
    response.raise_for_status()
    return json.loads(response.json()["message"]["content"])["messages"]


def generate_claude(client, text: str) -> list[str]:
    response = client.beta.messages.create(
        model=CLAUDE, max_tokens=4096, betas=["server-side-fallback-2026-07-01"],
        fallbacks="default", output_config={
            "effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": text}])
    if response.stop_reason == "refusal":
        return []
    body = next(b.text for b in response.content if b.type == "text")
    return json.loads(body)["messages"]


# Messages per (intent, language) and source: most from Qwen, a smaller group from Claude.
# (Qwen on the local GTX 1060 writes ~6 tokens/s: ~300 messages take about 2 hours.)
PER_LANGUAGE = {"qwen": {"es": 30, "pt": 18, "en": 12}, "claude": {"es": 10, "pt": 6, "en": 4}}

# Round 2: how people really type, for every intent (generated-<source>-r2.jsonl).
STYLE_ANGLES = [
    ("full of typing mistakes: missing or swapped letters, no accents, phone autocorrect "
     "errors, chat abbreviations (q, xq, pq, vc, pls)"),
    "regional slang (Mexico, Argentina, Colombia, Chile for Spanish; Brazil for Portuguese)",
    "mixing two languages in one message (Spanglish, Portuñol, English words)",
    "with emojis, or all in capital letters, or with lots of exclamation marks",
    "dictated by voice: no punctuation, run-on, filler words (oye, mira, então, like)",
    "long and rambling, with a personal story before the actual request",
    "extremely short: one to three words, maybe misspelt",
]
PER_LANGUAGE_R2 = {"qwen": {"es": 12, "pt": 8, "en": 5}, "claude": {"es": 6, "pt": 4, "en": 2}}


def generate(source: str, seed: int, round_: int = 1) -> None:
    rng = random.Random(seed + round_)
    out = DIR / (f"generated-{source}.jsonl" if round_ == 1 else f"generated-{source}-r2.jsonl")
    per_language = PER_LANGUAGE if round_ == 1 else PER_LANGUAGE_R2
    rows = read(out)
    done = {(r["intent"], r["language"]) for r in rows}
    client = None
    if source == "claude":
        import anthropic

        from aiplatform.config import get_settings
        key = get_settings().anthropic_api_key
        client = anthropic.Anthropic(api_key=key.get_secret_value() if key else None)
    for intent in INTENTS:
        for language, total in per_language[source].items():
            if (intent, language) in done:  # resumable: one batch per pair
                continue
            angles = (SPEC[intent][1] if round_ == 1 else STYLE_ANGLES)[:]
            rng.shuffle(angles)
            per_angle = max(2, round(total / len(angles)))
            batch: list[str] = []
            for angle in angles:
                if len(batch) >= total:
                    break
                text = prompt(intent, angle, language, per_angle)
                try:
                    messages = (generate_qwen(text) if source == "qwen"
                                else generate_claude(client, text))
                except (httpx.HTTPError, ValueError, KeyError) as exc:
                    print(f"  {intent}/{language}/{angle}: {type(exc).__name__}, skipped")
                    continue
                batch += [{"text": m.strip(), "angle": angle} for m in messages
                          if isinstance(m, str) and 2 <= len(m.strip()) <= 600]
            for i, item in enumerate(batch[:total]):
                prefix = source if round_ == 1 else f"{source}-r2"
                rows.append({"id": f"{prefix}-{intent}-{language}-{i:03d}", "text": item["text"],
                             "history": [], "intent": intent, "language": language,
                             "source": source, "angle": item["angle"]})
            write(out, rows)
            print(f"{source} {intent}/{language}: {len(batch[:total])}", flush=True)


# --- Training set ----------------------------------------------------------------------------

def embed(texts: list[str], batch: int = 64) -> list[list[float]]:
    vectors = []
    for i in range(0, len(texts), batch):
        response = httpx.post(f"{OLLAMA}/api/embed", timeout=600,
                              json={"model": EMBED_MODEL, "input": texts[i:i + batch]})
        response.raise_for_status()
        vectors += response.json()["embeddings"]
    return vectors


def build_train() -> list[dict]:
    import numpy as np

    test = read(TEST)
    files = [DIR / f"generated-{s}{r}.jsonl" for r in ("", "-r2") for s in ("qwen", "claude")]
    generated = [r for f in files for r in read(f)]
    # One misspelt copy of each generated message (seeded): typos the models learn from.
    generated += [{**r, "id": f"{r['id']}-typo", "text": add_typos(r["text"], Random(r["id"])),
                   "source": f"{r['source']}+typos"} for r in list(generated)]
    rows, seen = [], {normalized(r["text"]) for r in test}
    for r in generated:
        key = normalized(r["text"])
        if key and key not in seen:
            seen.add(key)
            rows.append(r)
    exact = len(generated) - len(rows)

    def unit(vectors):
        m = np.array(vectors, dtype=np.float32)
        return m / np.linalg.norm(m, axis=1, keepdims=True)
    similarity = unit(embed([r["text"] for r in rows])) @ unit(embed([r["text"] for r in test])).T
    keep = similarity.max(axis=1) < NEAR_DUPLICATE
    print(f"dropped {exact} exact and {int((~keep).sum())} near duplicates of test messages")
    return [r for r, k in zip(rows, keep) if k]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=["test", "generate", "train"])
    parser.add_argument("--source", choices=["qwen", "claude"], default="qwen")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--round", type=int, choices=[1, 2], default=1,
                        help="1: angles per intent; 2: typing styles (typos, slang, voice...)")
    args = parser.parse_args()
    if args.step == "test":
        rows = build_test()
        write(TEST, rows)
    elif args.step == "generate":
        generate(args.source, args.seed, args.round)
        return 0
    else:
        rows = build_train()
        write(TRAIN, rows)
    counts: dict = {}
    for r in rows:
        counts.setdefault(r["intent"], {}).setdefault(r["language"], 0)
        counts[r["intent"]][r["language"]] += 1
    print(f"{len(rows)} rows: {json.dumps(counts, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

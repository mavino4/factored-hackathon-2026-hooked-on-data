"""Intent classification: the step before the agent decides what to do with a message.

One short model call with a forced tool (``classify_intent``) says what the customer
wants, whether the agent can resolve it, and whether it needs the banking tools. It is
also where manipulation attempts (prompt injection) are caught: those get the ``attack``
intent and a fixed reply, and never reach the agent or its tools. If
the classifier fails or returns nothing usable, the turn falls back to the full agent
with tools (the behavior before classification existed).
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal

IntentName = Literal["account", "general", "out_of_scope", "human", "attack"]
INTENTS: tuple[IntentName, ...] = ("account", "general", "out_of_scope", "human", "attack")

CLASSIFY_TOOL: dict[str, Any] = {
    "name": "classify_intent",
    "description": "Classify the customer's last message.",
    "input_schema": {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": list(INTENTS)},
            "needs_tools": {"type": "boolean"},
            "insistence": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["intent", "needs_tools", "insistence", "reason"],
        "additionalProperties": False,
    },
}
FORCE_CLASSIFY = {"type": "tool", "name": "classify_intent"}

# How many recent messages the classifier reads, and how many repeats are insistence.
TRANSCRIPT_MESSAGES = 6
REPEATS_FOR_INSISTENCE = 3


@dataclass(frozen=True)
class Intent:
    name: IntentName
    needs_tools: bool
    insistence: bool
    reason: str
    fallback: bool = False  # the classifier gave no usable answer

    @property
    def summary(self) -> dict[str, Any]:
        return {"intent": self.name, "needs_tools": self.needs_tools,
                "insistence": self.insistence, "reason": self.reason,
                "fallback": self.fallback}


FALLBACK = Intent("account", needs_tools=True, insistence=False,
                  reason="classifier unavailable", fallback=True)


def is_customer_message(message: dict) -> bool:
    """Text the customer typed. User-role messages also carry tool results (a list) and
    the app's own notes (``author: system``, e.g. an approval decision): the author is set
    by the app, so nothing the customer types can pass for one."""
    return (message["role"] == "user" and isinstance(message["content"], str)
            and message.get("author") is None)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return " ".join(b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text")


def transcript(messages: list[dict], n: int = TRANSCRIPT_MESSAGES) -> str:
    """The last ``n`` customer/assistant texts. Tool calls and results and the app's own
    notes are left out: the classifier needs what was said, not account data."""
    lines = []
    for m in messages:
        text = _text(m["content"]).strip()
        if not text or m.get("author") == "system":
            continue
        who = "Customer" if m["role"] == "user" else (
            "Advisor" if m.get("author") == "operator" else "BankBot")
        lines.append(f"{who}: {text}")
    return "\n".join(lines[-n:])


def classify_messages(messages: list[dict]) -> list[dict]:
    """The single user message the classifier receives."""
    return [{"role": "user", "content": f"Conversation:\n{transcript(messages)}"}]


def parse(message) -> Intent:
    """The classifier's tool call as an Intent, or FALLBACK."""
    for block in message.content:
        if block.type == "tool_use" and block.name == "classify_intent":
            args = block.input if isinstance(block.input, dict) else {}
            name = args.get("intent")
            if name not in INTENTS:
                return FALLBACK
            return Intent(name, needs_tools=bool(args.get("needs_tools", name == "account")),
                          insistence=bool(args.get("insistence", False)),
                          reason=str(args.get("reason", ""))[:200])
    return FALLBACK


def _normalized(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^\w]+", " ", text).strip()


def repeated(messages: list[dict], times: int = REPEATS_FOR_INSISTENCE) -> bool:
    """The customer's last ``times`` messages say the same thing (deterministic backstop
    for insistence, whatever the classifier says)."""
    texts = [_normalized(m["content"]) for m in messages
             if is_customer_message(m) and m["content"].strip()]
    last = texts[-times:]
    return len(last) == times and len(set(last)) == 1 and bool(last[0])

"""Tool registry for agents. Each tool is a typed definition plus an async handler.

Handlers receive the model's arguments AND a server-side ToolContext. Identity (who
the user is) always comes from the context - the verified session - never from the
model's arguments, so a tool can't be steered into reading another user's data.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,), "integer": (int,), "number": (int, float),
    "boolean": (bool,), "object": (dict,), "array": (list,),
}


@dataclass(frozen=True)
class ToolContext:
    user_id: str  # verified token `sub` (or the dev-mode user)
    conversation_id: str | None = None


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any], ToolContext], Awaitable[Any]]
    # Irreversible actions (send, pay, delete) are never executed without human approval.
    irreversible: bool = False
    timeout_s: float = 30.0

    def definition(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "input_schema": self.input_schema}

    def validate(self, args: Any) -> str | None:
        """Return an error message if ``args`` doesn't match the top-level schema."""
        if not isinstance(args, dict):
            return "input must be an object"
        props = self.input_schema.get("properties", {})
        for key in self.input_schema.get("required", []):
            if key not in args:
                return f"missing required field: {key}"
        for key, value in args.items():
            if key not in props:
                return f"unexpected field: {key}"
            expected = _JSON_TYPES.get(props[key].get("type", ""))
            if expected and (not isinstance(value, expected)
                             or (isinstance(value, bool) and bool not in expected)):
                return f"field {key} must be {props[key]['type']}"
            if "enum" in props[key] and value not in props[key]["enum"]:
                return f"field {key} must be one of {props[key]['enum']}"
        return None

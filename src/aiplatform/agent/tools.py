"""Tool registry for agents. Each tool is a typed definition plus an async handler."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,), "integer": (int,), "number": (int, float),
    "boolean": (bool,), "object": (dict,), "array": (list,),
}


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Awaitable[str]]
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
        return None


# --- Example tools. Replace with real integrations. ---------------------------

async def _current_time(args: dict[str, Any]) -> str:
    return datetime.now(UTC).isoformat()


async def _create_ticket(args: dict[str, Any]) -> str:
    return f"Ticket created: {args['title']}"


DEFAULT_TOOLS: list[Tool] = [
    Tool(
        name="get_current_time",
        description="Get the current date and time in UTC (ISO 8601).",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=_current_time,
    ),
    Tool(
        name="create_support_ticket",
        description="Open a support ticket on behalf of the user. Use only after the user asks for it.",
        input_schema={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Short summary of the problem."},
                "details": {"type": "string", "description": "Full description."},
            },
            "required": ["title", "details"],
            "additionalProperties": False,
        },
        handler=_create_ticket,
        irreversible=True,
    ),
]

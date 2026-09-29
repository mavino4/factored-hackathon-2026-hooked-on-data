"""Events an agent run streams to the client."""

from dataclasses import dataclass
from typing import Any, Literal

Outcome = Literal["done", "approval_required", "refused", "truncated", "max_iterations"]


@dataclass
class AgentText:
    text: str


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResult:
    id: str
    name: str
    is_error: bool
    content: str


@dataclass
class ApprovalRequired:
    action_id: str
    tool_name: str
    input: dict[str, Any]


@dataclass
class AgentDone:
    outcome: Outcome
    text: str
    tool_calls: list[str]


AgentEvent = AgentText | ToolCall | ToolResult | ApprovalRequired | AgentDone

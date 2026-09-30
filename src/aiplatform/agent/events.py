"""Events an agent run streams to the client."""

from dataclasses import dataclass
from typing import Any, Literal

Outcome = Literal["done", "approval_required", "refused", "truncated", "max_iterations",
                  "handoff"]


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
class HandoffOffered:
    """The customer insists without being resolved: offer a human advisor (Yes / No)."""
    handoff_id: str


@dataclass
class HandoffStarted:
    """The conversation is now in the human queue; the bot stops answering."""
    handoff_id: str


@dataclass
class HumanWaiting:
    """A message arrived while a human handles the conversation: stored, not answered."""
    handoff_id: str


@dataclass
class AgentDone:
    outcome: Outcome
    text: str
    tool_calls: list[str]


AgentEvent = (AgentText | ToolCall | ToolResult | ApprovalRequired | HandoffOffered
              | HandoffStarted | HumanWaiting | AgentDone)

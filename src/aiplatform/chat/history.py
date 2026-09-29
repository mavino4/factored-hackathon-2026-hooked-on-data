"""Conversation history rules shared by chat turns and agent runs."""

# Claude Haiku 4.5 has a 200K-token context window; leave room for the reply.
MAX_HISTORY_CHARS = 150_000 * 4  # rough chars-per-token estimate


class ConversationTooLong(Exception):
    pass


def check_history_size(messages: list[dict]) -> None:
    if sum(len(str(m["content"])) for m in messages) > MAX_HISTORY_CHARS:
        raise ConversationTooLong()


def assistant_turn(message) -> dict:
    # Keep the full content blocks, not only the text.
    return {"role": "assistant", "content": [block.to_dict() for block in message.content]}

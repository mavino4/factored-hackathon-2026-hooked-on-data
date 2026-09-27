"""Database schema (SQLAlchemy Core). Alembic migrations in ``migrations/`` must match it."""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

metadata = sa.MetaData()

JSONType = sa.JSON().with_variant(JSONB(), "postgresql")

conversations = sa.Table(
    "conversations", metadata,
    sa.Column("id", sa.Uuid(as_uuid=False), primary_key=True),
    sa.Column("user_id", sa.Text, nullable=False),
    sa.Column("kind", sa.Text, nullable=False),
    sa.Column("title", sa.Text, nullable=True),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    sa.Index("ix_conversations_user_updated", "user_id", sa.text("updated_at DESC")),
)

# Append-only. The (conversation_id, seq) primary key rejects a second writer
# for the same position, so concurrent replicas can't interleave a history.
messages = sa.Table(
    "messages", metadata,
    sa.Column("conversation_id", sa.Uuid(as_uuid=False),
              sa.ForeignKey("conversations.id", ondelete="CASCADE"), primary_key=True),
    sa.Column("seq", sa.Integer, primary_key=True),
    sa.Column("role", sa.Text, nullable=False),
    sa.Column("content", JSONType, nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
)

usage_events = sa.Table(
    "usage_events", metadata,
    sa.Column("id", sa.BigInteger().with_variant(sa.Integer, "sqlite"),
              primary_key=True, autoincrement=True),
    sa.Column("user_id", sa.Text, nullable=False),
    sa.Column("conversation_id", sa.Uuid(as_uuid=False), nullable=True),
    sa.Column("route", sa.Text, nullable=False),
    sa.Column("provider", sa.Text, nullable=False),
    sa.Column("model", sa.Text, nullable=False),
    sa.Column("input_tokens", sa.Integer, nullable=False),
    sa.Column("output_tokens", sa.Integer, nullable=False),
    sa.Column("cache_read_tokens", sa.Integer, nullable=False),
    sa.Column("cache_write_tokens", sa.Integer, nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.Index("ix_usage_events_user_created", "user_id", "created_at"),
)

# Irreversible tool calls waiting for the user's decision. `status` moves from
# pending to approved/rejected exactly once (conditional UPDATE), so a tool can't run twice.
pending_actions = sa.Table(
    "pending_actions", metadata,
    sa.Column("id", sa.Uuid(as_uuid=False), primary_key=True),
    sa.Column("conversation_id", sa.Uuid(as_uuid=False),
              sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
    sa.Column("user_id", sa.Text, nullable=False),
    sa.Column("tool_use_id", sa.Text, nullable=False),
    sa.Column("tool_name", sa.Text, nullable=False),
    sa.Column("input", JSONType, nullable=False),
    sa.Column("status", sa.Text, nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
    sa.Index("ix_pending_actions_conversation", "conversation_id", "status"),
)

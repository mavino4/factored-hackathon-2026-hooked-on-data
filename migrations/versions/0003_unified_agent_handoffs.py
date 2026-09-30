"""One conversation type (agent), human handoffs, and message authors.

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-30
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Chat and agent are one flow now: former general-question chats become queries.
    op.execute("UPDATE conversations SET kind = 'agent' WHERE kind = 'chat'")
    op.add_column("messages", sa.Column("author", sa.Text, nullable=True))
    op.create_table(
        "handoffs",
        sa.Column("id", sa.Uuid(as_uuid=False), primary_key=True),
        sa.Column("conversation_id", sa.Uuid(as_uuid=False),
                  sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", sa.Text, nullable=False),
        sa.Column("status", sa.Text, nullable=False),
        sa.Column("reason", sa.Text, nullable=False),
        sa.Column("summary", sa.Text, nullable=False),
        sa.Column("message_index", sa.Integer, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_handoffs_conversation", "handoffs", ["conversation_id", "created_at"])
    op.create_index("ix_handoffs_status", "handoffs", ["status", "created_at"])


def downgrade() -> None:
    op.drop_table("handoffs")
    op.drop_column("messages", "author")
    # The chat/agent distinction is gone; converted conversations stay as queries.

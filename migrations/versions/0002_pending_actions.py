"""Pending actions awaiting user approval.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-27
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "pending_actions",
        sa.Column("id", sa.Uuid(as_uuid=False), primary_key=True),
        sa.Column("conversation_id", sa.Uuid(as_uuid=False),
                  sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", sa.Text, nullable=False),
        sa.Column("tool_use_id", sa.Text, nullable=False),
        sa.Column("tool_name", sa.Text, nullable=False),
        sa.Column("input", sa.JSON().with_variant(JSONB(), "postgresql"), nullable=False),
        sa.Column("status", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_pending_actions_conversation", "pending_actions",
                    ["conversation_id", "status"])


def downgrade() -> None:
    op.drop_table("pending_actions")

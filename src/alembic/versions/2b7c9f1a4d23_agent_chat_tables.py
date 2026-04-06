"""agent chat tables

Revision ID: 2b7c9f1a4d23
Revises: 765e5a082e76
Create Date: 2026-04-05

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


revision = "2b7c9f1a4d23"
down_revision = "765e5a082e76"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "agent_chat_conversation",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("thread_id", sqlmodel.sql.sqltypes.AutoString(length=255), nullable=False),
        sa.Column("title", sqlmodel.sql.sqltypes.AutoString(length=500), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "thread_id", name="uq_agent_chat_user_thread"),
    )
    op.create_index(
        op.f("ix_agent_chat_conversation_thread_id"),
        "agent_chat_conversation",
        ["thread_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_agent_chat_conversation_user_id"),
        "agent_chat_conversation",
        ["user_id"],
        unique=False,
    )

    op.create_table(
        "agent_chat_message",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("role", sqlmodel.sql.sqltypes.AutoString(length=32), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("trace_id", sqlmodel.sql.sqltypes.AutoString(length=255), nullable=True),
        sa.Column("extra", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["agent_chat_conversation.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_agent_chat_message_conversation_id"),
        "agent_chat_message",
        ["conversation_id"],
        unique=False,
    )


def downgrade():
    op.drop_index(
        op.f("ix_agent_chat_message_conversation_id"), table_name="agent_chat_message"
    )
    op.drop_table("agent_chat_message")
    op.drop_index(
        op.f("ix_agent_chat_conversation_user_id"), table_name="agent_chat_conversation"
    )
    op.drop_index(
        op.f("ix_agent_chat_conversation_thread_id"), table_name="agent_chat_conversation"
    )
    op.drop_table("agent_chat_conversation")

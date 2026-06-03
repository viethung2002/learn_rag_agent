"""Persistent agent chat history (user-facing), separate from LangGraph checkpoints."""

import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import JSON, Column, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, relationship
from sqlmodel import Field, Relationship, SQLModel


def _utc_now() -> datetime:
    return datetime.utcnow()


class AgentChatConversation(SQLModel, table=True):
    __tablename__ = "agent_chat_conversation"
    __table_args__ = (
        UniqueConstraint("user_id", "thread_id", name="uq_agent_chat_user_thread"),
    )

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: uuid.UUID = Field(foreign_key="user.id", ondelete="CASCADE", index=True)
    thread_id: str = Field(max_length=255, index=True)
    title: Optional[str] = Field(default=None, max_length=500)
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)

    messages: Mapped[list["AgentChatMessage"]] = Relationship(
        sa_relationship=relationship(
            "AgentChatMessage",
            back_populates="conversation",
            cascade="all, delete-orphan",
            passive_deletes=True,
        ),
    )


class AgentChatMessage(SQLModel, table=True):
    __tablename__ = "agent_chat_message"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    conversation_id: uuid.UUID = Field(
        foreign_key="agent_chat_conversation.id",
        ondelete="CASCADE",
        index=True,
    )
    role: str = Field(max_length=32, description="user | assistant")
    content: str = Field(sa_column=Column(Text, nullable=False))
    trace_id: Optional[str] = Field(default=None, max_length=255)
    extra: Optional[dict[str, Any]] = Field(
        default=None,
        sa_column=Column(JSON, nullable=True),
    )
    created_at: datetime = Field(default_factory=_utc_now)

    conversation: Mapped[Optional["AgentChatConversation"]] = Relationship(
        sa_relationship=relationship(
            "AgentChatConversation",
            back_populates="messages",
        )
    )

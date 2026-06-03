"""API schemas for persisted agent chat history."""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class AgentChatMessagePublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    role: str
    content: str
    trace_id: Optional[str] = None
    extra: Optional[dict[str, Any]] = None
    created_at: datetime


class AgentChatConversationPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    thread_id: str
    title: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class AgentChatConversationsResponse(BaseModel):
    data: list[AgentChatConversationPublic]
    total: int


class AgentChatMessagesResponse(BaseModel):
    thread_id: str
    messages: list[AgentChatMessagePublic]

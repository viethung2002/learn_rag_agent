"""CRUD for persisted agent chat (conversations + messages)."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import and_, func
from sqlmodel import Session, col, select

from src.models.agent_chat.schema import AgentChatConversation, AgentChatMessage


def _utc_now() -> datetime:
    return datetime.utcnow()


def get_or_create_conversation(
    session: Session,
    *,
    user_id: uuid.UUID,
    thread_id: str,
    title_hint: str,
) -> AgentChatConversation:
    stmt = select(AgentChatConversation).where(
        AgentChatConversation.user_id == user_id,
        AgentChatConversation.thread_id == thread_id,
    )
    conv = session.exec(stmt).first()
    if conv is not None:
        return conv
    preview = (title_hint or "").strip()
    title = preview[:500] if preview else None
    conv = AgentChatConversation(
        user_id=user_id,
        thread_id=thread_id,
        title=title,
        created_at=_utc_now(),
        updated_at=_utc_now(),
    )
    session.add(conv)
    session.flush()
    return conv


def append_turn(
    session: Session,
    *,
    user_id: uuid.UUID,
    thread_id: str,
    user_content: str,
    assistant_content: str,
    trace_id: Optional[str],
    extra: Optional[dict[str, Any]],
) -> None:
    conv = get_or_create_conversation(
        session,
        user_id=user_id,
        thread_id=thread_id,
        title_hint=user_content,
    )
    now = _utc_now()
    conv.updated_at = now
    session.add(conv)

    session.add(
        AgentChatMessage(
            conversation_id=conv.id,
            role="user",
            content=user_content,
            created_at=now,
        )
    )
    session.add(
        AgentChatMessage(
            conversation_id=conv.id,
            role="assistant",
            content=assistant_content,
            trace_id=trace_id,
            extra=extra,
            created_at=now,
        )
    )
    try:
        session.commit()
    except Exception:
        session.rollback()
        raise


def count_user_conversations(session: Session, *, user_id: uuid.UUID) -> int:
    stmt = select(func.count()).where(AgentChatConversation.user_id == user_id)
    return int(session.exec(stmt).one())


def list_conversations(
    session: Session,
    *,
    user_id: uuid.UUID,
    skip: int = 0,
    limit: int = 50,
) -> list[AgentChatConversation]:
    stmt = (
        select(AgentChatConversation)
        .where(AgentChatConversation.user_id == user_id)
        .order_by(col(AgentChatConversation.updated_at).desc())
        .offset(skip)
        .limit(limit)
    )
    return list(session.exec(stmt).all())


def get_conversation_by_thread(
    session: Session,
    *,
    user_id: uuid.UUID,
    thread_id: str,
) -> AgentChatConversation | None:
    stmt = select(AgentChatConversation).where(
        and_(
            AgentChatConversation.user_id == user_id,
            AgentChatConversation.thread_id == thread_id,
        )
    )
    return session.exec(stmt).first()


def list_messages_for_thread(
    session: Session,
    *,
    user_id: uuid.UUID,
    thread_id: str,
) -> list[AgentChatMessage] | None:
    conv = get_conversation_by_thread(
        session, user_id=user_id, thread_id=thread_id
    )
    if conv is None:
        return None
    stmt = (
        select(AgentChatMessage)
        .where(AgentChatMessage.conversation_id == conv.id)
        .order_by(col(AgentChatMessage.created_at).asc())
    )
    return list(session.exec(stmt).all())

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import func
from sqlmodel import Session, select, col


from src.core.security import get_password_hash, verify_password
from src.models.item.schema import ItemCreate,Item
from src.models.user.schema import User,UserCreate, UserUpdate
from src.models.agent_chat.schema import AgentChatConversation, AgentChatMessage

def create_user(*, session: Session, user_create: UserCreate) -> User:
    db_obj = User.model_validate(
        user_create, update={"hashed_password": get_password_hash(user_create.password)}
    )
    session.add(db_obj)
    session.commit()
    session.refresh(db_obj)
    return db_obj


def update_user(*, session: Session, db_user: User, user_in: UserUpdate) -> Any:
    user_data = user_in.model_dump(exclude_unset=True)
    extra_data = {}
    if "password" in user_data:
        password = user_data["password"]
        hashed_password = get_password_hash(password)
        extra_data["hashed_password"] = hashed_password
    db_user.sqlmodel_update(user_data, update=extra_data)
    session.add(db_user)
    session.commit()
    session.refresh(db_user)
    return db_user


def get_user_by_email(*, session: Session, email: str) -> User | None:
    statement = select(User).where(User.email == email)
    session_user = session.exec(statement).first()
    return session_user


def authenticate(*, session: Session, email: str, password: str) -> User | None:
    db_user = get_user_by_email(session=session, email=email)
    if not db_user:
        return None
    if not verify_password(password, db_user.hashed_password):
        return None
    return db_user


def create_item(*, session: Session, item_in: ItemCreate, owner_id: uuid.UUID) -> Item:
    db_item = Item.model_validate(item_in, update={"owner_id": owner_id})
    session.add(db_item)
    session.commit()
    session.refresh(db_item)
    return db_item


class AgentChatCRUD:
    def get_conversation(self, session: Session, user_id: uuid.UUID, thread_id: str) -> AgentChatConversation | None:
        statement = select(AgentChatConversation).where(
            AgentChatConversation.user_id == user_id,
            AgentChatConversation.thread_id == thread_id
        )
        return session.exec(statement).first()

    def get_conversation_by_thread_id(self, session: Session, thread_id: str) -> AgentChatConversation | None:
        statement = select(AgentChatConversation).where(
            AgentChatConversation.thread_id == thread_id
        )
        return session.exec(statement).first()

    def create_conversation(self, session: Session, user_id: uuid.UUID, thread_id: str, title: str | None = None) -> AgentChatConversation:
        db_obj = AgentChatConversation(user_id=user_id, thread_id=thread_id, title=title)
        session.add(db_obj)
        session.commit()
        session.refresh(db_obj)
        return db_obj

    def get_messages(self, session: Session, conversation_id: uuid.UUID) -> list[AgentChatMessage]:
        statement = select(AgentChatMessage).where(
            AgentChatMessage.conversation_id == conversation_id
        ).order_by(AgentChatMessage.created_at)
        return list(session.exec(statement).all())

    def list_messages_for_thread(self, session: Session, user_id: uuid.UUID, thread_id: str) -> list[AgentChatMessage] | None:
        """Return messages for a conversation identified by (user_id, thread_id).

        Returns None if no conversation exists for the given thread.
        """
        conv = self.get_conversation(session, user_id=user_id, thread_id=thread_id)
        if conv is None:
            return None
        statement = (
            select(AgentChatMessage)
            .where(AgentChatMessage.conversation_id == conv.id)
            .order_by(col(AgentChatMessage.created_at).asc())
        )
        return list(session.exec(statement).all())

    def create_message(self, session: Session, conversation_id: uuid.UUID, role: str, content: str, trace_id: str | None = None, extra: dict | None = None) -> AgentChatMessage:
        db_obj = AgentChatMessage(
            conversation_id=conversation_id,
            role=role,
            content=content,
            trace_id=trace_id,
            extra=extra or {}
        )
        session.add(db_obj)
        session.commit()
        session.refresh(db_obj)
        return db_obj

    def append_turn(
        self,
        session: Session,
        *,
        user_id: uuid.UUID,
        thread_id: str,
        user_content: str,
        assistant_content: str,
        trace_id: str | None,
        extra: dict | None,
    ) -> None:
        conv = self.get_conversation(session, user_id, thread_id)
        now = datetime.utcnow()
        if conv is None:
            preview = (user_content or "").strip()
            conv = AgentChatConversation(
                user_id=user_id,
                thread_id=thread_id,
                title=preview[:500] if preview else None,
                created_at=now,
                updated_at=now,
            )
            session.add(conv)
            session.flush()
        else:
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
        session.commit()

    def count_user_conversations(self, session: Session, *, user_id: uuid.UUID) -> int:
        statement = select(func.count()).where(AgentChatConversation.user_id == user_id)
        return int(session.exec(statement).one())

    def list_conversations(
        self,
        session: Session,
        *,
        user_id: uuid.UUID,
        skip: int = 0,
        limit: int = 50,
    ) -> list[AgentChatConversation]:
        statement = (
            select(AgentChatConversation)
            .where(AgentChatConversation.user_id == user_id)
            .order_by(AgentChatConversation.updated_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(session.exec(statement).all())

    def list_thread_ids(
        self,
        session: Session,
        *,
        user_id: uuid.UUID,
        skip: int = 0,
        limit: int = 100,
    ) -> list[str]:
        """Return a list of thread_id strings for a user's conversations (newest first)."""
        statement = (
            select(AgentChatConversation.thread_id)
            .where(AgentChatConversation.user_id == user_id)
            .order_by(AgentChatConversation.updated_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(session.exec(statement).all())

    def list_messages_for_thread(
        self,
        session: Session,
        *,
        user_id: uuid.UUID,
        thread_id: str,
    ) -> list[AgentChatMessage] | None:
        conv = self.get_conversation(session, user_id, thread_id)
        if conv is None:
            return None
        return self.get_messages(session, conv.id)

agent_chat = AgentChatCRUD()

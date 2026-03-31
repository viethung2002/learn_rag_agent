# app/core/base_schema.py
import uuid
from datetime import datetime
from pydantic import BaseModel, ConfigDict


class BaseSchema(BaseModel):
    """Schema cơ bản dùng cho tất cả API (có id, timestamps)"""
    id: uuid.UUID
    created_at: datetime | None = None
    updated_at: datetime | None = None
    deleted_at: datetime | None = None

    model_config = ConfigDict(
        orm_mode=True,          # Cho phép đọc từ SQLAlchemy/SQLModel
        from_attributes=True,   # Hỗ trợ convert từ ORM object
    )

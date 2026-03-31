# app/core/base_model.py
import uuid
from datetime import datetime
from sqlalchemy import Column, TIMESTAMP, func
from sqlmodel import SQLModel, Field
from sqlalchemy import  MetaData
from sqlalchemy.orm import declarative_base

# Naming convention chuẩn
metadata = MetaData(
    naming_convention={
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }
)

# Base dùng cho tất cả model
Base = declarative_base(metadata=metadata)


class BaseTime(SQLModel):
    """Mixin chứa các cột thời gian chuẩn: created_at, updated_at, deleted_at"""
    created_at: datetime | None = Field(
        sa_column=Column(
            TIMESTAMP(timezone=True),
            server_default=func.now(),
            nullable=False,
        ),
        description="Thời điểm tạo bản ghi",
    )

    updated_at: datetime | None = Field(
        sa_column=Column(
            TIMESTAMP(timezone=True),
            onupdate=func.now(),
            nullable=True,
        ),
        description="Thời điểm chỉnh sửa gần nhất",
    )

    deleted_at: datetime | None = Field(
        sa_column=Column(
            TIMESTAMP(timezone=True),
            nullable=True,
        ),
        description="Thời điểm xóa mềm (soft delete)",
    )


class Base_Model(BaseTime, SQLModel):
    """Mixin cơ bản cho tất cả bảng ORM"""
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)

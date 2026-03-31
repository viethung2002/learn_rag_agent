# src/models/paper_access/schema.py
from typing import Optional
import uuid
from sqlmodel import Field, SQLModel


class PaperBase(SQLModel):
    paper_id: str = Field(index=True, max_length=50)
    subject_type: str = Field(index=True, max_length=100)
    subject_id: str = Field(index=True, max_length=100)
    role: str = Field(index=True, max_length=50)
    session_id: str = Field(index=True, max_length=50)


class PaperAccess(PaperBase, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)

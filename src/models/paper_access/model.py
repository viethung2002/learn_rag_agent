# src/models/paper_access/model.py

import uuid
from sqlalchemy import Column, String
from src.db.interfaces.postgresql import Base


class PaperAccess(Base):
    __tablename__ = "paperaccess"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    paper_id = Column(String(50), index=True, nullable=False)
    subject_type = Column(String(100), index=True, nullable=False)
    subject_id = Column(String(100), index=True, nullable=False)
    role = Column(String(50), index=True, nullable=False, default="owner")
    session_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

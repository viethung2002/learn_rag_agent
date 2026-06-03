# src/repositories/paper_access.py

from typing import List
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.models.paper_access.model import PaperAccess


class PaperAccessRepository:
    def __init__(self, session: Session):
        self.session = session

    def grant_session_access(
        self,
        paper_id: str,
        session_id: str,
        subject_id: str, 
        role: str = "owner",
    ) -> PaperAccess:
        """
        Ghi một bản ghi ACL cho paper thuộc về một session cụ thể.

        - paper_id: id cua paper (arxiv_id)
        - subject_id: định danh của subject (ở đây là session_id)
        - session_id: định danh phiên của user (client gửi lên)
        - role: 'owner' | 'viewer' ... (tạm thời chỉ cần 'owner')
        """
        access = PaperAccess(
            paper_id=paper_id,
            subject_type="workspace",   # PHÂN BIỆT: quyền theo session
            subject_id=subject_id,
            role=role,
            session_id=session_id,
        )
        self.session.add(access)
        self.session.commit()
        self.session.refresh(access)
        return access

    def get_session_paper_ids(self, session_id: str) -> List[str]:
        """
        Lấy danh sách paper_id (arxiv_id) mà session này có quyền.
        """
        stmt = select(PaperAccess.paper_id).where(
            PaperAccess.subject_type == "session",
            PaperAccess.subject_id == session_id,
        )
        return list(self.session.scalars(stmt))

    def has_session_papers(self, session_id: str) -> bool:
        """
        Kiểm tra session này có ít nhất 1 paper được gán quyền hay không.
        """
        stmt = select(func.count(PaperAccess.id)).where(
            PaperAccess.subject_type == "session",
            PaperAccess.subject_id == session_id,
        )
        return (self.session.scalar(stmt) or 0) > 0

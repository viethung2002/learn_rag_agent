"""Sync Postgres ``Paper`` rows into Neo4j (Paper nodes; optional Author + WROTE)."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from src.models.paper import Paper
from src.services.neo4j.client import Neo4jClient

logger = logging.getLogger(__name__)

# Merge paper by business key; update scalar fields on each run.
_MERGE_PAPER = """
MERGE (p:Paper {arxiv_id: $arxiv_id})
SET p.paper_id = $paper_id,
    p.title = $title,
    p.abstract = $abstract,
    p.pdf_url = $pdf_url,
    p.published_date = $published_date,
    p.categories = $categories,
    p.pdf_processed = $pdf_processed
"""

# Link authors after paper exists.
_LINK_AUTHORS = """
MATCH (p:Paper {arxiv_id: $arxiv_id})
WITH p
UNWIND $author_names AS author_name
MERGE (a:Author {name: author_name})
MERGE (a)-[:WROTE]->(p)
"""

# --- BƯỚC 1: THÊM CÂU LỆNH TẠO ĐỒ THỊ TRÍCH DẪN ---
_LINK_REFERENCES = """
MATCH (p:Paper {arxiv_id: $arxiv_id})
WITH p
UNWIND $references AS ref_title
// Tạo Node Reference với title làm định danh
MERGE (r:Reference {title: ref_title})
// Nối bài báo gốc CITES (trích dẫn) Reference này
MERGE (p)-[:CITES]->(r)
"""
# --------------------------------------------------


def _paper_to_merge_params(paper: Any) -> Dict[str, Any]:
    # Dùng getattr để lấy dữ liệu an toàn
    authors = getattr(paper, "authors", []) or []
    if not isinstance(authors, list):
        authors = []

    categories = getattr(paper, "categories", []) or []
    if not isinstance(categories, list):
        categories = []

    published = getattr(paper, "published_date", None)
    
    if isinstance(published, str):
        published_str = published
    elif hasattr(published, "isoformat"):
        published_str = published.isoformat()
    else:
        published_str = str(published) if published is not None else ""

    if hasattr(paper, "id") and paper.id:
        pid = str(paper.id)
    else:
        pid = str(getattr(paper, "arxiv_id", ""))

    is_processed = getattr(paper, "pdf_processed", False)

# --- BƯỚC 2: XỬ LÝ LỌC TÊN BÀI BÁO TRÍCH DẪN ---
    raw_refs = getattr(paper, "references", []) or []
    if not isinstance(raw_refs, list):
        raw_refs = []
        
    # LOG TRINH SÁT: Kiểm tra xem Neo4j có nhận được dữ liệu thô từ DB không
    logger.info(f"DEBUG NEO4J: Đã nhận {len(raw_refs)} raw_refs từ DB cho bài báo {pid}")
        
    extracted_refs = []
    title_pattern = re.compile(r"['\"‘“](.*?)['\"’”]")
    
    for r in raw_refs:
        raw_text = ""
        if isinstance(r, dict) and "raw_text" in r:
            raw_text = r["raw_text"]
        elif isinstance(r, str):
            raw_text = r
            
        if raw_text:
            match = title_pattern.search(raw_text)
            if match:
                title = match.group(1).strip()
                title = title.rstrip(",.")
                if len(title) > 15:
                    extracted_refs.append(title)
            else:
                # --- NẾU BẠN THIẾU ĐOẠN ELSE NÀY, KẾT QUẢ SẼ LUÔN LÀ 0 ---
                title = raw_text.strip()
                title = re.sub(r"^\s*\[\d+\]\s*", "", title)
                title = re.sub(r"^\s*\d+\.\s*", "", title)
                if len(title) > 15:
                    extracted_refs.append(title[:250])  # Lấy tối đa 250 ký tự

    logger.info(f"DEBUG NEO4J: Tách được {len(extracted_refs)} trích dẫn cho bài báo {pid}")
    # -----------------------------------------------

    return {
        "arxiv_id": getattr(paper, "arxiv_id", ""),
        "paper_id": pid,
        "title": getattr(paper, "title", "") or "",
        "abstract": getattr(paper, "abstract", "") or "",
        "pdf_url": getattr(paper, "pdf_url", "") or "",
        "published_date": published_str,
        "categories": [str(c) for c in categories],
        "pdf_processed": bool(is_processed),
        "author_names": [str(a).strip() for a in authors if str(a).strip()],
        "references": extracted_refs, # Trả mảng title đã lọc về cho Neo4j
    }


class PaperGraphIngestion:
    """Upsert papers into Neo4j from SQLAlchemy ``Paper`` instances."""

    def __init__(self, neo4j_client: Neo4jClient, *, link_authors: bool = True):
        self._client = neo4j_client
        self._link_authors = link_authors

    def ingest_paper(self, paper: Paper) -> None:
        params = _paper_to_merge_params(paper)
        
        # Bóc 2 mảng này ra để chạy Cypher riêng
        author_names = params.pop("author_names", [])
        references = params.pop("references", []) # Lấy mảng references ra
        
        # 1. Tạo Paper Node
        self._client.execute_write(_MERGE_PAPER, params)
        
        # 2. Link Authors
        if self._link_authors and author_names:
            self._client.execute_write(
                _LINK_AUTHORS,
                {"arxiv_id": params["arxiv_id"], "author_names": author_names},
            )
            
        # --- BƯỚC 3: THỰC THI LINK REFERENCES ---
        if references:
            self._client.execute_write(
                _LINK_REFERENCES,
                {"arxiv_id": params["arxiv_id"], "references": references},
            )
        # ----------------------------------------

    def ingest_papers(self, papers: List[Paper]) -> Dict[str, int]:
        ok = 0
        for paper in papers:
            try:
                self.ingest_paper(paper)
                ok += 1
            except Exception as e:
                logger.error("Neo4j ingest failed for %s: %s", getattr(paper, "arxiv_id", "Unknown"), e)
        return {"ingested": ok, "total": len(papers)}

    def ingest_from_session(
        self,
        session: Session,
        *,
        limit: int = 500,
        offset: int = 0,
        processed_only: bool = False,
    ) -> Dict[str, int]:
        q = session.query(Paper).order_by(Paper.updated_at.desc())
        if processed_only:
            q = q.filter(Paper.pdf_processed.is_(True))
        rows = q.offset(offset).limit(limit).all()
        stats = self.ingest_papers(rows)
        stats["offset"] = offset
        stats["limit"] = limit
        return stats

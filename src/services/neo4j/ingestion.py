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

# --- CẬP NHẬT: GỘP QUERY THÀNH BATCH VỚI UNWIND ---
_BATCH_MERGE_PAPER = """
UNWIND $papers AS p_data
MERGE (p:Paper {arxiv_id: p_data.arxiv_id})
SET p.paper_id = p_data.paper_id,
    p.title = p_data.title,
    p.abstract = p_data.abstract,
    p.pdf_url = p_data.pdf_url,
    p.published_date = p_data.published_date,
    p.categories = p_data.categories,
    p.pdf_processed = p_data.pdf_processed
"""

_BATCH_LINK_AUTHORS = """
UNWIND $papers AS p_data
MATCH (p:Paper {arxiv_id: p_data.arxiv_id})
UNWIND p_data.author_names AS author_name
MERGE (a:Author {name: author_name})
MERGE (a)-[:WROTE]->(p)
"""

_BATCH_LINK_REFERENCES = """
UNWIND $papers AS p_data
MATCH (p:Paper {arxiv_id: p_data.arxiv_id})
UNWIND p_data.references AS ref_title
MERGE (r:Reference {title: ref_title})
MERGE (p)-[:CITES]->(r)
"""

# --- THÊM ĐOẠN NÀY VÀO ĐÂY ---
_RESOLVE_CITATIONS_QUERY = """
MATCH (p1:Paper)-[:CITES]->(r:Reference)
MATCH (p2:Paper)
WHERE toLower(p2.title) = toLower(r.title) OR toLower(p2.title) CONTAINS toLower(r.title)
MERGE (p1)-[:CITES_PAPER]->(p2)
RETURN count(p2) as matched_count
"""
# --------------------------------------------------

def _paper_to_merge_params(paper: Any) -> Dict[str, Any]:
    # ... [GIỮ NGUYÊN HOÀN TOÀN HÀM NÀY CỦA BẠN] ...
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

    raw_refs = getattr(paper, "references", []) or []
    if not isinstance(raw_refs, list):
        raw_refs = []
        
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
                title = raw_text.strip()
                title = re.sub(r"^\s*\[\d+\]\s*", "", title)
                title = re.sub(r"^\s*\d+\.\s*", "", title)
                if len(title) > 15:
                    extracted_refs.append(title[:250])

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
        "references": extracted_refs,
    }


class PaperGraphIngestion:
    """Upsert papers into Neo4j from SQLAlchemy ``Paper`` instances."""

    def __init__(self, neo4j_client: Neo4jClient, *, link_authors: bool = True):
        self._client = neo4j_client
        self._link_authors = link_authors

    def ingest_papers(self, papers: List[Paper], batch_size: int = 100) -> Dict[str, int]:
        """Tối ưu hóa: Ingest nhiều bài báo cùng lúc thay vì chạy vòng lặp từng bài."""
        if not papers:
            return {"ingested": 0, "total": 0}

        total_ingested = 0
        
        # Xử lý theo từng batch (lô) để tránh query quá to
        for i in range(0, len(papers), batch_size):
            batch = papers[i:i + batch_size]
            
            # Chuyển đổi dữ liệu
            params_list = []
            for p in batch:
                try:
                    params_list.append(_paper_to_merge_params(p))
                except Exception as e:
                    logger.error("Error parsing paper for Neo4j: %s", e)
            
            if not params_list:
                continue

            try:
                # 1. Batch Tạo Paper Node
                self._client.execute_write(_BATCH_MERGE_PAPER, {"papers": params_list})
                
                # 2. Batch Link Authors
                if self._link_authors:
                    self._client.execute_write(_BATCH_LINK_AUTHORS, {"papers": params_list})
                    
                # 3. Batch Link References (Chỉ gửi những bài có reference lên)
                refs_list = [p for p in params_list if p.get("references")]
                if refs_list:
                    self._client.execute_write(_BATCH_LINK_REFERENCES, {"papers": refs_list})
                    
                total_ingested += len(params_list)
                logger.info(f"Neo4j: Successfully ingested batch of {len(params_list)} papers")
            except Exception as e:
                logger.error("Neo4j batch ingest failed: %s", e)

        return {"ingested": total_ingested, "total": len(papers)}

    def ingest_paper(self, paper: Paper) -> None:
        """Giữ lại để đảm bảo tương thích ngược (backward compatibility) với các file cũ"""
        self.ingest_papers([paper])

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
            
        # TỐI ƯU RAM: Dùng yield_per thay vì .all()
        q = q.offset(offset).limit(limit)
        
        batch = []
        total_ingested = 0
        
        for paper in q.yield_per(50):
            batch.append(paper)
            if len(batch) >= 100: # Gom đủ 100 bài thì nhét vào Neo4j 1 lần
                stats = self.ingest_papers(batch)
                total_ingested += stats["ingested"]
                batch.clear() # Xóa bộ nhớ lô cũ
                
        # Ingest số còn dư (nếu có)
        if batch:
            stats = self.ingest_papers(batch)
            total_ingested += stats["ingested"]

        return {"ingested": total_ingested, "total": limit, "offset": offset, "limit": limit}
    
    def resolve_internal_citations(self) -> int:
            """
            Quét đồ thị và nối các node Reference ảo thành liên kết CITES_PAPER 
            giữa các bài báo đã có thật trong hệ thống.
            """
            try:
                result = self._client.execute_write(_RESOLVE_CITATIONS_QUERY)
                # Neo4j client trả về list of dicts, lấy giá trị của 'matched_count'
                matched = result[0]["matched_count"] if result else 0
                logger.info(f"Neo4j: Resolved and linked {matched} internal citations.")
                return matched
            except Exception as e:
                logger.error(f"Neo4j citation resolution failed: {e}")
                return 0

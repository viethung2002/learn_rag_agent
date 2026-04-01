"""Sync Postgres ``Paper`` rows into Neo4j (Paper nodes; optional Author + WROTE)."""

from __future__ import annotations

import logging
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

# Link authors after paper exists (requires unique Author.name if you use constraint).
_LINK_AUTHORS = """
MATCH (p:Paper {arxiv_id: $arxiv_id})
WITH p
UNWIND $author_names AS author_name
MERGE (a:Author {name: author_name})
MERGE (a)-[:WROTE]->(p)
"""


def _paper_to_merge_params(paper: Paper) -> Dict[str, Any]:
    authors = paper.authors or []
    if not isinstance(authors, list):
        authors = []

    categories = paper.categories or []
    if not isinstance(categories, list):
        categories = []

    published = paper.published_date
    published_str = published.isoformat() if published is not None else ""

    pid: str
    if isinstance(paper.id, UUID):
        pid = str(paper.id)
    else:
        pid = str(paper.id)

    return {
        "arxiv_id": paper.arxiv_id,
        "paper_id": pid,
        "title": paper.title or "",
        "abstract": paper.abstract or "",
        "pdf_url": paper.pdf_url or "",
        "published_date": published_str,
        "categories": [str(c) for c in categories],
        "pdf_processed": bool(paper.pdf_processed),
        "author_names": [str(a).strip() for a in authors if str(a).strip()],
    }


class PaperGraphIngestion:
    """Upsert papers into Neo4j from SQLAlchemy ``Paper`` instances."""

    def __init__(self, neo4j_client: Neo4jClient, *, link_authors: bool = True):
        self._client = neo4j_client
        self._link_authors = link_authors

    def ingest_paper(self, paper: Paper) -> None:
        params = _paper_to_merge_params(paper)
        author_names = params.pop("author_names")
        self._client.execute_write(_MERGE_PAPER, params)
        if self._link_authors and author_names:
            self._client.execute_write(
                _LINK_AUTHORS,
                {"arxiv_id": params["arxiv_id"], "author_names": author_names},
            )

    def ingest_papers(self, papers: List[Paper]) -> Dict[str, int]:
        ok = 0
        for paper in papers:
            try:
                self.ingest_paper(paper)
                ok += 1
            except Exception as e:
                logger.error("Neo4j ingest failed for %s: %s", paper.arxiv_id, e)
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

import asyncio
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from arxiv_ingestion.common import get_pdf_parser_service, get_opensearch_service, get_db_service,get_arxiv_service




logger = logging.getLogger(__name__)


def _raw_text_lines(pdf_content) -> list[str]:
    if not pdf_content.raw_text:
        return []
    return [line.strip() for line in pdf_content.raw_text.split("\n") if line.strip()]


def _first_substantial_title_line(lines: list[str], *, max_scan: int = 15) -> str | None:
    """Skip arXiv header lines; prefer the first long line that looks like a paper title."""
    for line in lines[:max_scan]:
        if "arxiv:" in line.lower():
            continue
        if len(line) > 30:
            return line
    return None


def _title_line_index(lines: list[str], title: str) -> int | None:
    if not lines or not title or title == "Untitled Paper":
        return None
    for i, line in enumerate(lines[:25]):
        if line == title:
            return i
        if len(title) > 25 and title in line:
            return i
    # Title may differ slightly from raw line; align to first substantial non-arXiv line
    for i, line in enumerate(lines[:15]):
        if "arxiv:" in line.lower():
            continue
        if len(line) > 30:
            return i
    return None


def extract_metadata_from_pdf(pdf_content):
    """Extract title, authors, abstract, categories from parsed PDF content.

    Title and authors come from raw text line order. Docling section titles are not used for
    the paper title because they often duplicate author names (e.g. \"Name & Name\").
    """
    title = "Untitled Paper"
    authors = ["Unknown Author"]
    abstract = "No abstract available"
    categories = []

    lines = _raw_text_lines(pdf_content)

    # 1) Title: raw text only — first long line after skipping arXiv banners (not Docling sections)
    if lines:
        picked = _first_substantial_title_line(lines)
        if picked:
            title = picked
        elif len(lines) > 1 and "arxiv:" in lines[0].lower():
            title = lines[1]
        else:
            title = lines[0]
        if "arxiv:" in title.lower() or (
            re.search(r"arxiv:\s*\d{4}\.\d+", title, re.IGNORECASE) and len(title) < 120
        ):
            picked2 = _first_substantial_title_line(lines)
            if picked2:
                title = picked2

    # 2) Categories: arXiv-style [cs.CV] in brackets, then loose tokens
    if pdf_content.raw_text:
        head = pdf_content.raw_text[:3000]
        cat_match = re.findall(r"\[([a-z][a-z0-9-]*\.[A-Z][A-Za-z0-9-]*)\]", head)
        if cat_match:
            categories = list(dict.fromkeys(cat_match))
        else:
            fallback_cats = re.findall(
                r"\b(cs\.[A-Z]{2}|math\.[A-Z]{2}|stat\.[A-Z]{2}|eess\.[A-Z]{2}|physics\.[a-z-]+|q-bio\.[A-Z]{2})\b",
                head,
            )
            if fallback_cats:
                categories = list(dict.fromkeys(fallback_cats))

        if not categories and pdf_content.metadata:
            if "subject" in pdf_content.metadata:
                subject = pdf_content.metadata.get("subject", "")
                if isinstance(subject, str) and "." in subject:
                    categories = [s.strip() for s in subject.split(",") if s.strip()]
            elif "keywords" in pdf_content.metadata:
                keywords = pdf_content.metadata.get("keywords", "")
                if isinstance(keywords, str):
                    for kw in keywords.split(","):
                        kw = kw.strip()
                        if re.match(r"^[a-z]+\.[A-Z]+$", kw):
                            categories.append(kw)

    # 3) Abstract: section first, then "Abstract - ..." / "Abstract:" heuristics
    for section in pdf_content.sections or []:
        if section.title and "abstract" in section.title.lower():
            abstract = (section.content or "").strip()[:1500]
            break

    if abstract == "No abstract available" and pdf_content.raw_text:
        rt = pdf_content.raw_text
        abs_match = re.search(
            r"Abstract\s*-\s*(.*?)(?:I\.\s+INTRODUCTION|Keywords|1\.\s+Introduction)",
            rt,
            re.DOTALL | re.IGNORECASE,
        )
        if abs_match:
            abstract = re.sub(r"\s+", " ", abs_match.group(1).strip())[:1500]
        else:
            for pattern in (
                r"(?:^|\n)\s*Abstract\s*:?\s*\n\s*(.*?)(?:\n\s*(?:1\.|Introduction|Keywords|Categories|I\.|§|Chapter|Section)|$)",
                r"Abstract\s*:?\s+(.*?)(?:\n\s*(?:1\.|Introduction|Keywords|Categories|I\.|§)|$)",
                r"abstract[:\s]+(.*?)(?:introduction|1\.|keywords|categories|§|chapter)",
            ):
                match = re.search(pattern, rt[:5000], re.IGNORECASE | re.DOTALL | re.MULTILINE)
                if match:
                    abstract_text = re.sub(r"\s+", " ", match.group(1).strip())
                    if len(abstract_text) > 50:
                        abstract = abstract_text[:1500]
                        break

    # 4) Authors: line immediately after title in raw text (split on comma, &, and)
    if lines:
        idx = _title_line_index(lines, title)
        if idx is not None and idx + 1 < len(lines):
            author_line = lines[idx + 1]
            if "arxiv:" not in author_line.lower() and len(author_line) < 400:
                if "@" in author_line or "university" in author_line.lower():
                    authors = ["Multiple Authors"]
                else:
                    author_list = re.split(r",|&|\s+and\s+", author_line, flags=re.IGNORECASE)
                    authors = [a.strip() for a in author_list if len(a.strip()) > 2]

    if not categories:
        categories = ["cs.AI"]

    t_preview = title[:80] + ("…" if len(title) > 80 else "")
    logger.info(
        "Extracted metadata - title: %r, abstract length: %s, categories: %s",
        t_preview,
        len(abstract),
        categories,
    )

    return {
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "categories": categories,
    }


def process_user_uploaded_paper(**context):
    from src.repositories.paper_access import PaperAccessRepository
    from src.repositories.paper import PaperRepository
    """Process a user-uploaded PDF paper: parse, extract metadata, store, and index.
    
    Expects arxiv_id and pdf_path in DAG run conf.
    """
    ti = context.get("ti")
    dag_run = context.get("dag_run")
    
    conf = dag_run.conf if dag_run else {}
    arxiv_id = conf.get("arxiv_id")
    pdf_path_str = conf.get("pdf_path")
    session_id = conf.get("session_id") 
    
    if not arxiv_id:
        raise ValueError("arxiv_id must be provided in DAG run conf")
    if not pdf_path_str:
        raise ValueError("pdf_path must be provided in DAG run conf")
    if not session_id:
        raise ValueError("session_id must be provided in DAG run conf (required for grant_session_access)")

    # Convert to Airflow container path if needed
    pdf_path = Path(pdf_path_str)
    # If path is from API container, convert to Airflow container path
    if "/app/data/user_uploads" in str(pdf_path):
        pdf_path = Path(str(pdf_path).replace("/app/data/user_uploads", "/opt/airflow/user_uploads"))
    
    if not pdf_path.exists():
        raise ValueError(f"PDF file not found: {pdf_path}")
    
    logger.info(f"Processing user-uploaded paper: arxiv_id={arxiv_id}, pdf_path={pdf_path}")
    
    arxiv_client= get_arxiv_service()
    pdf_parser= get_pdf_parser_service()
    database=get_db_service()
    opensearch_client= get_opensearch_service()
    
    with database.get_session() as session:
        paper_repo = PaperRepository(session)
        access_repo = PaperAccessRepository(session)

        # 1) Kiểm tra nếu paper đã tồn tại và đã processed
        existing_paper = paper_repo.get_by_arxiv_id(arxiv_id)

        if existing_paper and existing_paper.pdf_processed:
            # Signature: grant_session_access(paper_id, session_id, subject_id, role=...)
            access_repo.grant_session_access(arxiv_id, session_id, session_id)
            logger.info(f"Paper {arxiv_id} already processed, skipping")
            if ti:
                ti.xcom_push(key="status", value="already_processed")
                ti.xcom_push(key="paper_id", value=str(existing_paper.id))
            return {"status": "already_processed", "paper_id": str(existing_paper.id)}

        # 2) Nếu chưa processed: parse + lưu + index
        result = asyncio.run(
            _parse_extract_and_index(
                existing_paper, arxiv_id, pdf_path, pdf_parser, opensearch_client, session, paper_repo
            )
        )

        access_repo.grant_session_access(arxiv_id, session_id, session_id)

        if ti:
            ti.xcom_push(key="status", value=result.get("status"))
            ti.xcom_push(key="paper_id", value=result.get("paper_id"))
            ti.xcom_push(key="chunks_indexed", value=result.get("chunks_indexed", 0))

        return result


async def _parse_extract_and_index(paper, arxiv_id, pdf_path, pdf_parser, opensearch_client, session, paper_repo):
    from src.schemas.arxiv.paper import PaperCreate
    """Parse PDF, extract metadata, create/update paper, and index."""
    from arxiv_ingestion.indexing import _index_papers_with_chunks
    
    try:
        # 1. Parse PDF
        logger.info(f"--- BẮT ĐẦU DEBUG DOCLING CHO {arxiv_id} ---")
        pdf_content = await pdf_parser.parse_pdf(pdf_path)

        if not pdf_content:
            raise ValueError(f"Failed to parse PDF: {pdf_path}")

        # --- ĐOẠN DEBUG QUAN TRỌNG ---
        raw_text_preview = pdf_content.raw_text[:2000] if pdf_content.raw_text else "EMPTY"
        logger.info(f"[RAW TEXT PREVIEW]:\n{raw_text_preview}")

        if pdf_content.sections:
            logger.info(f"[SECTIONS DETECTED]: {[s.title for s in pdf_content.sections]}")

        logger.info(f"[PDF METADATA]: {pdf_content.metadata}")
        logger.info(f"--- KẾT THÚC DEBUG DOCLING ---")

        metadata = extract_metadata_from_pdf(pdf_content)
        logger.info(f"Extracted metadata: title='{metadata['title']}', authors={metadata['authors']}, categories={metadata['categories']}")
        
        # Create or update paper
        if paper:
            # Update existing paper
            logger.info(f"Updating existing paper {arxiv_id}")
            paper.title = metadata["title"]
            paper.authors = metadata["authors"]
            paper.abstract = metadata["abstract"]
            paper.categories = metadata["categories"]
            paper.raw_text = pdf_content.raw_text
            paper.sections = [s.model_dump() for s in pdf_content.sections] if pdf_content.sections else None
            
            # SỬA TẠI ĐÂY: Convert list string sang list dict cho references
            if pdf_content.references:
                paper.references = [{"raw_content": r} if isinstance(r, str) else r for r in pdf_content.references]
            else:
                paper.references = None
                
            paper.parser_used = "DOCLING"
            paper.parser_metadata = pdf_content.metadata
            paper.pdf_processed = True
            paper.pdf_processing_date = datetime.now(timezone.utc)
            paper = paper_repo.update(paper)
        else:
            # Create new paper
            logger.info(f"Creating new paper {arxiv_id}")
            
            # SỬA TẠI ĐÂY: Chuẩn bị dữ liệu references cho PaperCreate
            formatted_references = None
            if pdf_content.references:
                formatted_references = [{"raw_content": r} if isinstance(r, str) else r for r in pdf_content.references]

            paper_data = PaperCreate(
                arxiv_id=arxiv_id,
                title=metadata["title"],
                authors=metadata["authors"],
                abstract=metadata["abstract"],
                categories=metadata["categories"],
                published_date=datetime.now(timezone.utc),
                pdf_url=f"user-upload://{arxiv_id}",
                raw_text=pdf_content.raw_text,
                sections=[s.model_dump() for s in pdf_content.sections] if pdf_content.sections else None,
                references=formatted_references, # Sử dụng dữ liệu đã format
                parser_used="docling",
                parser_metadata=pdf_content.metadata,
                pdf_processed=True,
                pdf_processing_date=datetime.now(timezone.utc),
            )
            paper = paper_repo.create(paper_data)
        
        session.commit()
        logger.info(f"Paper {arxiv_id} stored: title='{metadata['title']}', authors={metadata['authors']}, categories={metadata['categories']}")
        logger.info(f"Dữ liệu trích dẫn chuẩn bị đẩy: {pdf_content.references}")
        try:
            from arxiv_ingestion.common import get_neo4j_service
            from src.services.neo4j.ingestion import PaperGraphIngestion

            logger.info(f"Đang đẩy bài báo {arxiv_id} lên Neo4j...")
            neo4j_client = get_neo4j_service()
            graph_ingest = PaperGraphIngestion(neo4j_client)
            graph_ingest.ingest_paper(paper)
            graph_ingest.resolve_internal_citations()
            logger.info(f"Đã đồng bộ bài báo {arxiv_id} lên Neo4j thành công.")
        except Exception as e:
            logger.error(f"Lỗi khi đẩy dữ liệu lên Neo4j: {e}", exc_info=True)

        # Index in OpenSearch using the same function as arxiv_paper_ingestion
        logger.info(f"Indexing paper {arxiv_id} in OpenSearch using _index_papers_with_chunks")
        indexing_stats = await _index_papers_with_chunks([paper])
        
        logger.info(
            f"Successfully processed paper {arxiv_id}: "
            f"{indexing_stats.get('total_chunks_indexed', 0)} chunks indexed, "
            f"{indexing_stats.get('total_chunks_created', 0)} chunks created"
        )
        
        return {
            "status": "success",
            "paper_id": str(paper.id),
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "authors": paper.authors,
            "chunks_indexed": indexing_stats.get("total_chunks_indexed", 0),
            "chunks_created": indexing_stats.get("total_chunks_created", 0),
        }
    except Exception as e:
        logger.error(f"Error processing paper {arxiv_id}: {e}", exc_info=True)
        raise

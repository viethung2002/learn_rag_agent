import asyncio
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from arxiv_ingestion.common import get_cached_services

from src.repositories.paper_access import PaperAccessRepository
from src.repositories.paper import PaperRepository
from src.schemas.arxiv.paper import PaperCreate

logger = logging.getLogger(__name__)


def extract_metadata_from_pdf(pdf_content):
    """Extract title, authors, abstract, categories from parsed PDF content.
    
    Args:
        pdf_content: PdfContent object from parser
        
    Returns:
        dict with title, authors, abstract, categories
    """
    title = "Untitled Paper"
    authors = ["Unknown Author"]
    abstract = "No abstract available"
    categories = []
    
    # Extract title from first section or raw text
    if pdf_content.sections:
        first_section = pdf_content.sections[0]
        if first_section.title and first_section.title.lower() not in ["content", "introduction", "abstract"]:
            title = first_section.title
        elif first_section.content:
            first_line = first_section.content.split("\n")[0].strip()
            if len(first_line) > 10 and len(first_line) < 200:
                title = first_line
    elif pdf_content.raw_text:
        lines = [line.strip() for line in pdf_content.raw_text.split("\n") if line.strip()]
        if lines:
            potential_title = lines[0]
            if len(potential_title) > 10 and len(potential_title) < 200:
                title = potential_title
    
    # Extract abstract - try multiple approaches
    # First, try to find "Abstract:" section
    for section in pdf_content.sections:
        if section.title and "abstract" in section.title.lower():
            abstract = section.content.strip()[:1000]
            break
    
    # If no abstract section, search in raw text for "Abstract:" pattern
    if abstract == "No abstract available" and pdf_content.raw_text:
        # Look for "Abstract:" or "ABSTRACT:" followed by text
        abstract_patterns = [
            # Pattern 1: "Abstract:" or "ABSTRACT:" on its own line, followed by content
            r"(?:^|\n)\s*Abstract\s*:?\s*\n\s*(.*?)(?:\n\s*(?:1\.|Introduction|Keywords|Categories|I\.|§|Chapter|Section)|$)",
            # Pattern 2: "Abstract" followed by colon and content
            r"Abstract\s*:?\s+(.*?)(?:\n\s*(?:1\.|Introduction|Keywords|Categories|I\.|§)|$)",
            # Pattern 3: More flexible pattern
            r"abstract[:\s]+(.*?)(?:introduction|1\.|keywords|categories|§|chapter)",
        ]
        
        for pattern in abstract_patterns:
            match = re.search(pattern, pdf_content.raw_text[:5000], re.IGNORECASE | re.DOTALL | re.MULTILINE)
            if match:
                abstract_text = match.group(1).strip()
                # Clean up - remove extra whitespace and newlines
                abstract_text = re.sub(r'\s+', ' ', abstract_text)
                if len(abstract_text) > 50:  # Make sure we got meaningful content
                    abstract = abstract_text[:1000]
                    break
    
    # Extract categories - look for "Categories:" pattern
    if pdf_content.raw_text:
        # Improved category patterns - look for "Categories:" specifically
        category_patterns = [
            # Pattern 1: "Categories:" on its own line, followed by categories
            r"(?:^|\n)\s*Categories?\s*:?\s*\n\s*([A-Za-z0-9\.\s,]+?)(?:\n\s*(?:Abstract|Keywords|Subject|1\.|Introduction)|$)",
            # Pattern 2: "Categories:" followed by categories on same or next line
            r"Categories?\s*:?\s+([A-Za-z0-9\.\s,]+?)(?:\n\s*(?:Abstract|Keywords|Subject|1\.|Introduction)|$)",
            # Pattern 3: More flexible
            r"categories?[:\s]+([A-Za-z0-9\.\s,]+?)(?:\n|;|\.|keywords|subject|abstract|1\.)",
            # Pattern 4: Subject Classification
            r"subject\s+classification[:\s]+([A-Za-z0-9\.\s,]+?)(?:\n|;|\.|keywords)",
            # Pattern 5: ACM Classification
            r"acm\s+classification[:\s]+([A-Za-z0-9\.\s,]+?)(?:\n|;|\.)",
        ]
        
        for pattern in category_patterns:
            match = re.search(pattern, pdf_content.raw_text[:3000], re.IGNORECASE | re.DOTALL | re.MULTILINE)
            if match:
                categories_text = match.group(1).strip()
                logger.info(f"Found categories text: {categories_text}")
                
                # Clean up - remove extra whitespace
                categories_text = re.sub(r'\s+', ' ', categories_text)
                
                # Parse categories - could be comma-separated or space-separated
                if "," in categories_text:
                    categories = [c.strip() for c in categories_text.split(",") if c.strip()]
                else:
                    # Try to split by common separators
                    categories = re.split(r"[;\s]+", categories_text)
                    categories = [c.strip() for c in categories if c.strip() and len(c.strip()) < 50]
                
                # Filter to only valid arXiv-style categories (e.g., "cs.CV", "cs.AI", "math.OC")
                valid_categories = []
                for cat in categories:
                    cat = cat.strip()
                    # Check if it looks like arXiv category (e.g., "cs.CV", "math.OC", "eess.IV")
                    # Pattern: lowercase letters, dot, uppercase letters
                    if re.match(r"^[a-z]+\.[A-Z]+$", cat):
                        valid_categories.append(cat)
                    # Or if it's a common pattern with dot
                    elif "." in cat and len(cat.split(".")) == 2:
                        parts = cat.split(".")
                        if parts[0].islower() and parts[1].isupper():
                            valid_categories.append(cat)
                
                if valid_categories:
                    categories = valid_categories
                    logger.info(f"Extracted categories: {categories}")
                    break
        
        # Also check PDF metadata
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
    
    # Default categories if none found
    if not categories:
        categories = ["user-upload"]
    
    # Extract authors - typically at the beginning, before abstract
    if pdf_content.raw_text:
        text_start = pdf_content.raw_text[:1500]
        author_lines = []
        for line in text_start.split("\n")[:10]:
            line = line.strip()
            if line and len(line) < 200:
                if " and " in line.lower() or ("," in line and len(line.split(",")) <= 5):
                    author_lines.append(line)
        
        if author_lines:
            authors_text = author_lines[0]
            if "," in authors_text:
                authors = [a.strip() for a in authors_text.split(",")]
            elif " and " in authors_text.lower():
                parts = re.split(r"\s+and\s+", authors_text, flags=re.IGNORECASE)
                authors = [p.strip() for p in parts]
            else:
                authors = [authors_text]
    
    logger.info(f"Extracted metadata - title: '{title[:50]}...', abstract length: {len(abstract)}, categories: {categories}")
    
    return {
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "categories": categories,
    }


def process_user_uploaded_paper(**context):
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
    
    # Convert to Airflow container path if needed
    pdf_path = Path(pdf_path_str)
    # If path is from API container, convert to Airflow container path
    if "/app/data/user_uploads" in str(pdf_path):
        pdf_path = Path(str(pdf_path).replace("/app/data/user_uploads", "/opt/airflow/user_uploads"))
    
    if not pdf_path.exists():
        raise ValueError(f"PDF file not found: {pdf_path}")
    
    logger.info(f"Processing user-uploaded paper: arxiv_id={arxiv_id}, pdf_path={pdf_path}")
    
    arxiv_client, pdf_parser, database, metadata_fetcher, opensearch_client = get_cached_services()
    
    with database.get_session() as session:
        paper_repo = PaperRepository(session)
        access_repo = PaperAccessRepository(session)

        # 1) Kiểm tra nếu paper đã tồn tại và đã processed
        existing_paper = paper_repo.get_by_arxiv_id(arxiv_id)

        if existing_paper and existing_paper.pdf_processed:
            # Gắn ACL cho session này với paper đã có
            access_repo.grant_session_access(
                paper_id=arxiv_id,      # dùng chính arxiv_id làm key
                session_id=session_id,
                role="owner",
            )
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

        # Sau khi _parse_extract_and_index tạo/cập nhật Paper thành công,
        # gắn ACL cho session này:
        access_repo.grant_session_access(
            paper_id=arxiv_id,
            session_id=session_id,
            role="owner",
        )

        if ti:
            ti.xcom_push(key="status", value=result.get("status"))
            ti.xcom_push(key="paper_id", value=result.get("paper_id"))
            ti.xcom_push(key="chunks_indexed", value=result.get("chunks_indexed", 0))

        return result


async def _parse_extract_and_index(paper, arxiv_id, pdf_path, pdf_parser, opensearch_client, session, paper_repo):
    """Parse PDF, extract metadata, create/update paper, and index."""
    from arxiv_ingestion.indexing import _index_papers_with_chunks
    
    try:
        # Parse PDF
        logger.info(f"Parsing PDF: {pdf_path}")
        pdf_content = await pdf_parser.parse_pdf(pdf_path)
        
        if not pdf_content:
            raise ValueError(f"Failed to parse PDF: {pdf_path}")
        
        logger.info(f"PDF parsed successfully: {len(pdf_content.raw_text) if pdf_content.raw_text else 0} chars extracted")
        
        # Extract metadata from PDF
        logger.info("Extracting metadata from PDF...")
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
            paper.references = [r for r in pdf_content.references] if pdf_content.references else None
            paper.parser_used = "DOCLING"
            paper.parser_metadata = pdf_content.metadata
            paper.pdf_processed = True
            paper.pdf_processing_date = datetime.now(timezone.utc)
            paper = paper_repo.update(paper)
        else:
            # Create new paper
            logger.info(f"Creating new paper {arxiv_id}")
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
                references=[r for r in pdf_content.references] if pdf_content.references else None,
                parser_used="docling",
                parser_metadata=pdf_content.metadata,
                pdf_processed=True,
                pdf_processing_date=datetime.now(timezone.utc),
            )
            paper = paper_repo.create(paper_data)
        
        session.commit()
        logger.info(f"Paper {arxiv_id} stored: title='{metadata['title']}', authors={metadata['authors']}, categories={metadata['categories']}")
        
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

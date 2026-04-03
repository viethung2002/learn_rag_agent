import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

from .common import get_arxiv_service, get_db_service, get_metadata_fetcher_service

logger = logging.getLogger(__name__)


async def run_paper_ingestion_pipeline(
    target_date: str,
    process_pdfs: bool = True,
) -> dict:
    """Async wrapper for the paper ingestion pipeline.

    :param target_date: Date to fetch papers for (YYYYMMDD format)
    :param process_pdfs: Whether to download and process PDFs
    :returns: Dictionary with ingestion statistics
    """
    arxiv_client = get_arxiv_service()
    database = get_db_service()
    metadata_fetcher = get_metadata_fetcher_service()


    max_results = arxiv_client.max_results
    logger.info(f"Using default max_results from config: {max_results}")

    with database.get_session() as session:
        return await metadata_fetcher.fetch_and_process_papers(
            max_results=max_results,
            from_date=target_date,
            to_date=target_date,
            process_pdfs=process_pdfs,
            store_to_db=True,
            db_session=session,
        )


def fetch_daily_papers(**context):
    """Fetch daily papers from arXiv and store in PostgreSQL.

    This task:
    1. Determines the target date (defaults to yesterday)
    2. Fetches papers from arXiv API
    3. Downloads and processes PDFs using Docling
    4. Stores metadata and parsed content in PostgreSQL

    Note: OpenSearch indexing is handled by a separate dedicated task
    """
    logger.info("Starting daily paper fetching task")

    execution_date = context.get("execution_date")
    if execution_date:
        target_dt = execution_date - timedelta(days=3)
        target_date = target_dt.strftime("%Y%m%d")
    else:
        yesterday = datetime.now() - timedelta(days=3)
        target_date = yesterday.strftime("%Y%m%d")

    logger.info(f"Fetching papers for date: {target_date}")

    results = asyncio.run(
        run_paper_ingestion_pipeline(
            target_date=target_date,
            process_pdfs=True,
        )
    )

    logger.info(f"Daily fetch complete: {results['papers_fetched']} papers for {target_date}")

    results["date"] = target_date
    ti = context.get("ti")
    if ti:
        ti.xcom_push(key="fetch_results", value=results)

    return results


def resolve_citations(**context):
    """Task to resolve and link internal citations in Neo4j.
    
    This task runs after daily fetching to connect newly ingested papers
    with existing references in the knowledge graph.
    """
    logger.info("Starting internal citation resolution task")
    
    # Import lười (Lazy import) để không nạp Neo4j client khi không cần thiết
    from .common import get_neo4j_service
    from src.services.neo4j.ingestion import PaperGraphIngestion

    try:
        neo4j_client = get_neo4j_service()
        ingestion = PaperGraphIngestion(neo4j_client)
        
        # Gọi hàm hợp nhất mà bạn vừa viết bên file ingestion.py
        matched = ingestion.resolve_internal_citations()
        
        logger.info(f"Successfully resolved and linked {matched} internal citations")
        
        # Lưu kết quả vào XCom nếu muốn hiển thị ra Report sau này
        ti = context.get("ti")
        if ti:
            ti.xcom_push(key="citation_results", value={"matched_citations": matched})
            
        return {"matched_citations": matched}
        
    except Exception as e:
        logger.error(f"Failed to resolve citations: {e}")
        raise

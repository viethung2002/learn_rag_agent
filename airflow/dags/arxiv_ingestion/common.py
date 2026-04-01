import logging
import sys
from functools import lru_cache
from typing import Any, Tuple

sys.path.insert(0, "/opt/airflow")

from src.config import get_settings
from src.db.factory import make_database
from src.services.arxiv.factory import make_arxiv_client
from src.services.metadata_fetcher import make_metadata_fetcher
from src.services.neo4j.factory import make_neo4j_client
from src.services.neo4j.ingestion import PaperGraphIngestion
from src.services.opensearch.factory import make_opensearch_client
from src.services.pdf_parser.factory import make_pdf_parser_service

logger = logging.getLogger(__name__)

def get_cached_services() -> Tuple[Any, Any, Any, Any, Any]:
    """Initialize and return service instances."""
    logger.info("Initializing services")

    settings = get_settings()

    arxiv_client = make_arxiv_client()
    pdf_parser = make_pdf_parser_service()
    database = make_database()
    opensearch_client = make_opensearch_client()

    neo4j_client = make_neo4j_client(settings)
    neo4j_ingestion = PaperGraphIngestion(neo4j_client)

    metadata_fetcher = make_metadata_fetcher(
        arxiv_client,
        pdf_parser,
        neo4j_ingestion=neo4j_ingestion,
    )

    logger.info("All services initialized successfully")

    return arxiv_client, pdf_parser, database, metadata_fetcher, opensearch_client

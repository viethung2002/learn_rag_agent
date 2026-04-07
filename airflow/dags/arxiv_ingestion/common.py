import logging
import sys
from functools import lru_cache

sys.path.insert(0, "/opt/airflow")

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_db_service():
    logger.info("Initializing Database service")
    from src.db.factory import make_database

    return make_database()


@lru_cache(maxsize=1)
def get_arxiv_service():
    logger.info("Initializing Arxiv client")
    from src.services.arxiv.factory import make_arxiv_client

    return make_arxiv_client()


@lru_cache(maxsize=1)
def get_pdf_parser_service():
    logger.info("Initializing PDF Parser (Heavy Service)")
    from src.services.pdf_parser.factory import make_pdf_parser_service

    return make_pdf_parser_service()


@lru_cache(maxsize=1)
def get_opensearch_service():
    logger.info("Initializing OpenSearch client")
    from src.services.opensearch.factory import make_opensearch_client

    return make_opensearch_client()


@lru_cache(maxsize=1)
def get_neo4j_service():
    logger.info("Initializing Neo4j client")
    from src.config import get_settings
    from src.services.neo4j.factory import make_neo4j_client

    settings = get_settings()
    return make_neo4j_client(settings)


@lru_cache(maxsize=1)
def get_metadata_fetcher_service():
    logger.info("Initializing Metadata Fetcher")

    # Local imports to avoid heavy startup cost during DAG parsing
    from src.services.metadata_fetcher import make_metadata_fetcher
    from src.services.neo4j.ingestion import PaperGraphIngestion

    # Use other cached getters which will perform their own lazy imports
    arxiv_client = get_arxiv_service()
    pdf_parser = get_pdf_parser_service()
    neo4j_client = get_neo4j_service()
    neo4j_ingestion = PaperGraphIngestion(neo4j_client)

    return make_metadata_fetcher(
        arxiv_client,
        pdf_parser,
        neo4j_ingestion=neo4j_ingestion,
    )

import logging
from sqlalchemy import text

# CHỈ import 2 hàm cần thiết, xóa bỏ get_cached_services và các service thừa
from .common import get_db_service, get_opensearch_service

logger = logging.getLogger(__name__)

def setup_environment():
    """Setup environment and verify dependencies.
    Creates hybrid search index with RRF pipeline.
    """
    logger.info("Setting up environment for arXiv paper ingestion")

    try:
        database = get_db_service()
        opensearch_client = get_opensearch_service()

        with database.get_session() as session:
            session.execute(text("SELECT 1"))
            logger.info("Database connection verified")

        try:
            health = opensearch_client.client.cluster.health()
            if health["status"] in ["green", "yellow", "red"]:
                logger.info(f"OpenSearch hybrid client connected (cluster status: {health['status']})")
            else:
                raise Exception(f"OpenSearch cluster unhealthy: {health['status']}")
        except Exception as e:
            raise Exception(f"OpenSearch hybrid client connection failed: {e}")

        setup_results = opensearch_client.setup_indices(force=False)
        if setup_results.get("hybrid_index"):
            logger.info("Hybrid search index created with vector support")
        else:
            logger.info("Hybrid search index already exists")

        if setup_results.get("rrf_pipeline"):
            logger.info("RRF pipeline created successfully")
        else:
            logger.info("RRF pipeline already exists")

        logger.info("Hybrid search setup completed")

        return {"status": "success", "message": "Environment setup completed"}

    except Exception as e:
        error_msg = f"Environment setup failed: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

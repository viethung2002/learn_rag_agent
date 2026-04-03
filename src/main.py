import logging
import os
from pathlib import Path
from click import command
import sentry_sdk
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi import FastAPI
from fastapi.routing import APIRoute
from starlette.middleware.cors import CORSMiddleware

from src.api.main import api_router
from src.core.config import settings

from src.config import get_settings
from src.db.factory import make_database
from src.routers import agentic_ask, hybrid_search, ping
from src.routers.ask import ask_router, stream_router
from src.routers.ask_gemini import ask_gemini,stream_gemini
from src.routers.ask_nvidia import ask_nvidia,stream_nvidia
from src.routers import agentic_ask, hybrid_search, ping, upload

from src.services.arxiv.factory import make_arxiv_client
from src.services.cache.factory import make_cache_client
from src.services.embeddings.factory import make_embeddings_service
from src.services.langfuse.factory import make_langfuse_tracer
from src.services.ollama.factory import make_ollama_client
from src.services.gemini.factory import make_gemini_client
from src.services.nvidia.factory import make_nvidia_client
from src.services.opensearch.factory import make_opensearch_client
from src.services.neo4j.factory import make_neo4j_client, make_neo4j_driver

from src.services.pdf_parser.factory import make_pdf_parser_service
from src.services.telegram.factory import make_telegram_service
from src.services.agents.factory import make_agentic_rag_service


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def custom_generate_unique_id(route: APIRoute) -> str:
    return f"{route.tags[0]}-{route.name}"
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan for the API.
    """
    logger.info("Starting RAG API...")
    try:
        alembic_cfg = uvicorn.Config(str(Path(__file__).resolve().parent / "alembic.ini"))
        command.upgrade(alembic_cfg, "head")
        from sqlmodel import Session
        from src.core.db import engine, init_db
        with Session(engine) as session:
            init_db(session)
        logger.info("Database migrations and initial user ready")
    except Exception as e:
        logger.warning("DB migration/init non-fatal: %s", e)

    settings = get_settings()
    app.state.settings = settings
   
    database = make_database()
    app.state.database = database
    logger.info("Database connected")

    # Initialize search service
    opensearch_client = make_opensearch_client()
    app.state.opensearch_client = opensearch_client

    neo4j_client = make_neo4j_client(settings)
    app.state.neo4j_client = neo4j_client
    try:
        neo4j_client.verify_connectivity()
        logger.info("Neo4j connected successfully")
    except Exception as e:
        logger.warning("Neo4j connectivity check failed - graph features may be limited: %s", e)

    # Verify OpenSearch connectivity and create index if needed
    if opensearch_client.health_check():
        logger.info("OpenSearch connected successfully")
        

        # Setup hybrid index (supports all search types)
        setup_results = opensearch_client.setup_indices(force=False)
        if setup_results.get("hybrid_index"):
            logger.info("Hybrid index created")
        else:
            logger.info("Hybrid index already exists")

        # Get simple statistics
        try:
            stats = opensearch_client.client.count(index=opensearch_client.index_name)
            logger.info(f"OpenSearch ready: {stats['count']} documents indexed")
        except Exception:
            logger.info("OpenSearch index ready (stats unavailable)")
    else:
        logger.warning("OpenSearch connection failed - search features will be limited")

    # Initialize other services (kept for future endpoints and notebook demos)
    app.state.arxiv_client = make_arxiv_client()
    app.state.pdf_parser = make_pdf_parser_service()
    app.state.embeddings_service = make_embeddings_service()
    app.state.ollama_client = make_ollama_client()
    app.state.gemini_client = make_gemini_client()
    app.state.nvidia_client = make_nvidia_client()
    app.state.langfuse_tracer = make_langfuse_tracer()
    app.state.cache_client = make_cache_client(settings)
    logger.info("Services initialized: arXiv API client, PDF parser, OpenSearch, Embeddings, Ollama, Langfuse, Cache")

    # Initialize Telegram bot (Week 7)
    telegram_service = make_telegram_service(
        opensearch_client=app.state.opensearch_client,
        embeddings_client=app.state.embeddings_service,
        nvidia_client=app.state.nvidia_client,
        cache_client=app.state.cache_client,
        langfuse_tracer=app.state.langfuse_tracer,
    )

    if telegram_service:
        app.state.telegram_service = telegram_service
        try:
            await telegram_service.start()
            logger.info("Telegram bot started successfully")
        except Exception as e:
            logger.error(f"Failed to start Telegram bot: {e}")
    else:
        logger.info("Telegram bot not configured - skipping initialization")
    
    # Initialize Agentic
    app.state.agentic_rag = make_agentic_rag_service(
        opensearch_client=app.state.opensearch_client,
        nvidia_client=app.state.nvidia_client,
        embeddings_client=app.state.embeddings_service,
        langfuse_tracer=app.state.langfuse_tracer,
    )

    logger.info("API ready")
    yield

    # Cleanup
    if hasattr(app.state, "telegram_service") and app.state.telegram_service:
        await app.state.telegram_service.stop()
        logger.info("Telegram bot stopped")

    if getattr(app.state, "neo4j_client", None) is not None:
        try:
            app.state.neo4j_client.close()
            logger.info("Neo4j client closed")
        except Exception as e:
            logger.warning("Error closing Neo4j client: %s", e)
        make_neo4j_driver.cache_clear()

    database.teardown()
    logger.info("API shutdown complete")

if settings.SENTRY_DSN and settings.ENVIRONMENT != "local":
    sentry_sdk.init(dsn=str(settings.SENTRY_DSN), enable_tracing=True)
app = FastAPI(
    title="arXiv Paper Curator API",
    description="Personal arXiv CS.AI paper curator with RAG capabilities",
    version=os.getenv("APP_VERSION", "0.1.0"),
    openapi_url=f"{settings.API_V1_STR}/openapi.json",\
    generate_unique_id_function=custom_generate_unique_id,
    lifespan=lifespan,
)
if settings.all_cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.all_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
# Include routers
app.include_router(ping.router, prefix="/api/v1")  # Health check endpoint
app.include_router(hybrid_search.router, prefix="/api/v1")  # Search chunks with BM25/hybrid
app.include_router(ask_router, prefix="/api/v1")  # RAG question answering with LLM
app.include_router(stream_router, prefix="/api/v1")  # Streaming RAG responses
app.include_router(ask_gemini, prefix="/api/v1")  # RAG question answering with Gemini LLM
app.include_router(stream_gemini, prefix="/api/v1")  # Streaming RAG
app.include_router(ask_nvidia, prefix="/api/v1")  # RAG question answering with Nvidia LLM
app.include_router(stream_nvidia, prefix="/api/v1")  # Streaming RAG
app.include_router(api_router, prefix=settings.API_V1_STR)

app.include_router(agentic_ask.router)  # Agentic RAG with intelligent retrieval
app.include_router(upload.router, prefix="/api/v1")  # Paper upload endpoint


if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")

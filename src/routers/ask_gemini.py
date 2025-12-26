import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from src.dependencies import EmbeddingsDep, GeminiDep, OpenSearchDep
from src.schemas.api.ask import AskRequest, AskResponse

logger = logging.getLogger(__name__)

# Two separate routers - one for regular ask, one for streaming
ask_router_api = APIRouter(tags=["ask"])
stream_router_api = APIRouter(tags=["stream"])


async def _prepare_chunks_and_sources(
    request: AskRequest,
    opensearch_client,
    embeddings_service,
):
    """Shared function to prepare chunks and sources for RAG."""
    # Generate query embedding for hybrid search if enabled
    query_embedding = None
    search_mode = "bm25"

    if request.use_hybrid:
        try:
            query_embedding = await embeddings_service.embed_query(request.query)
            search_mode = "hybrid"
            logger.info("Generated query embedding for hybrid search")
        except Exception as e:
            logger.warning(f"Failed to generate embeddings, falling back to BM25: {e}")
            query_embedding = None
            search_mode = "bm25"

    # Retrieve top-k chunks
    logger.info(f"Retrieving top {request.top_k} chunks for query: '{request.query}'")

    search_results = opensearch_client.search_unified(
        query=request.query,
        query_embedding=query_embedding,
        size=request.top_k,
        from_=0,
        categories=request.categories,
        use_hybrid=request.use_hybrid and query_embedding is not None,
        min_score=0.0,
    )

    # Extract chunks with minimal data for LLM
    chunks = []
    sources_set = set()  # Use set to automatically handle duplicates

    for hit in search_results.get("hits", []):
        arxiv_id = hit.get("arxiv_id", "")

        # Build minimal chunk for LLM (only content + arxiv_id)
        chunk_data = {
            "arxiv_id": arxiv_id,
            "chunk_text": hit.get("chunk_text", hit.get("abstract", "")),
        }
        chunks.append(chunk_data)

        # Build PDF URL from arxiv_id for sources (automatically deduplicates)
        if arxiv_id:
            arxiv_id_clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf"
            sources_set.add(pdf_url)

    # Convert set back to list for consistent return type
    sources = list(sources_set)

    return chunks, sources, search_mode


@stream_router_api.post("/stream_gemini/stream", response_model=AskResponse)
async def stream_ask_gemini(
    request: AskRequest,
    opensearch_client: OpenSearchDep = OpenSearchDep,
    embeddings_service: EmbeddingsDep = EmbeddingsDep,
    gemini_client = GeminiDep, 
):
    """Streamed endpoint to ask questions to Gemini with RAG."""
    try:
        chunks, sources, search_mode = await _prepare_chunks_and_sources(
            request,
            opensearch_client,
            embeddings_service,
        )
    except Exception as e:
        logger.error(f"Error preparing chunks and sources: {e}")
        raise HTTPException(status_code=500, detail="Error preparing data for the request.")

    async def event_generator():
        """Generator to stream events back to the client."""
        try:
            async for chunk in gemini_client.stream_gemini_response(
                query=request.query,
                chunks=chunks,
                model=request.model,
            ):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"

            # Send final metadata
            yield f"data: {json.dumps({'done': True, 'answer': chunk, 'sources': sources})}\n\n"

        except Exception as e:
            logger.error(f"Error during streaming response: {e}")
            yield f"data: {json.dumps({'error': 'Internal server error during response generation.'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )

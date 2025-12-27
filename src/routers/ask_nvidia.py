import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from src.dependencies import EmbeddingsDep, NvidiaDep, OpenSearchDep
from src.schemas.api.ask import AskRequest, AskResponse

logger = logging.getLogger(__name__)

# Two separate routers - one for regular ask, one for streaming
ask_nvidia = APIRouter(tags=["ask"])
stream_nvidia = APIRouter(tags=["stream"])


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


@ask_nvidia.post("/ask_nvidia", response_model=AskResponse)
async def ask_nvidia_api(
    request: AskRequest,
    opensearch_client: OpenSearchDep,
    embeddings_service: EmbeddingsDep,
    nvidia_client: NvidiaDep,
):
    """Endpoint to ask questions to nvidia with RAG."""
    try:
        chunks, sources, search_mode = await _prepare_chunks_and_sources(
            request,
            opensearch_client,
            embeddings_service,
        )
    except Exception as e:
        logger.error(f"Error preparing chunks and sources: {e}")
        raise HTTPException(status_code=500, detail="Error preparing data for the request.")

    if not chunks:
        logger.warning("No chunks retrieved for the query")
        # Có thể trả về câu trả lời mặc định hoặc lỗi nhẹ
        return AskResponse(
            query=request.query,
            answer="Không tìm thấy tài liệu liên quan để trả lời câu hỏi của bạn.",
            sources=[],
            chunks_used=0,
            search_mode=search_mode,
        )

    try:
        rag_result = await nvidia_client.generate_rag_answer(
            query=request.query,
            chunks=chunks,
            model=request.model or None,  # nếu model=None thì dùng default trong client
            use_structured_output=True,
        )

    except Exception as e:
        logger.error(f"Error getting response from nvidia: {e}")
        raise HTTPException(status_code=500, detail="Error generating response from nvidia.")

    # Trích xuất và đảm bảo các field đúng định dạng cho AskResponse
    return AskResponse(
        query=request.query,
        answer=rag_result.get("answer", "Không thể tạo câu trả lời."),  # phải là str
        sources=rag_result.get("sources", sources),  # ưu tiên từ nvidia, fallback về sources từ search
        chunks_used=len(chunks),
        search_mode=search_mode,
    )


@stream_nvidia.post("/stream_nvidia/stream", response_model=AskResponse)
async def stream_ask_nvidia(
    request: AskRequest,
    opensearch_client: OpenSearchDep ,
    embeddings_service: EmbeddingsDep,
    nvidia_client: NvidiaDep,
)-> StreamingResponse:
    """Streamed endpoint to ask questions to nvidia with RAG."""
    async def generate_stream():
        try:
            if not opensearch_client.health_check():
                yield f"data: {json.dumps({'error': 'Search service unavailable'})}\n\n"
                return

            await nvidia_client.health_check()

            # Get chunks and sources using shared function
            chunks, sources, search_mode = await _prepare_chunks_and_sources(request, opensearch_client, embeddings_service)

            if not chunks:
                yield f"data: {json.dumps({'answer': 'No relevant information found.', 'sources': [], 'done': True})}\n\n"
                return

            # Send metadata first
            yield f"data: {json.dumps({'sources': sources, 'chunks_used': len(chunks), 'search_mode': search_mode})}\n\n"

            # Stream the answer
            full_response = ""
            async for chunk in nvidia_client.generate_rag_answer_stream(query=request.query, chunks=chunks, model=request.model):
                if chunk.get("response"):
                    text_chunk = chunk["response"]
                    full_response += text_chunk
                    yield f"data: {json.dumps({'chunk': text_chunk})}\n\n"

                if chunk.get("done", False):
                    yield f"data: {json.dumps({'answer': full_response, 'done': True})}\n\n"
                    break

        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate_stream(), media_type="text/plain", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

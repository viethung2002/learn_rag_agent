import json
import logging
import time
from typing import Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from src.dependencies import CacheDep, EmbeddingsDep, LangfuseDep, NvidiaDep, OpenSearchDep
from src.schemas.api.ask import AskRequest, AskResponse
from src.services.langfuse.tracer import RAGTracer

logger = logging.getLogger(__name__)

# Two separate routers - one for regular ask, one for streaming
ask_nvidia = APIRouter(tags=["ask"])
stream_nvidia = APIRouter(tags=["stream"])


async def _prepare_chunks_and_sources(
    request: AskRequest,
    opensearch_client,
    embeddings_service,
    rag_tracer: RAGTracer,
    trace=None,
) -> tuple[List[Dict], List[str], str]:
    """Shared function to prepare chunks and sources with tracing."""

    query_embedding = None
    search_mode = "bm25"

    # Trace embedding generation if hybrid search is enabled
    if request.use_hybrid:
        with rag_tracer.trace_embedding(trace, request.query) as embedding_span:
            try:
                query_embedding = await embeddings_service.embed_query(request.query)
                search_mode = "hybrid"
                logger.info("Generated query embedding for hybrid search")
            except Exception as e:
                logger.warning(f"Failed to generate embeddings, falling back to BM25: {e}")
                if embedding_span:
                    rag_tracer.tracer.update_span(embedding_span, output={"success": False, "error": str(e)})
                query_embedding = None
                search_mode = "bm25"

    # Trace search step
    with rag_tracer.trace_search(trace, request.query, request.top_k) as search_span:
        search_results = opensearch_client.search_unified(
            query=request.query,
            query_embedding=query_embedding,
            size=request.top_k,
            from_=0,
            categories=request.categories,
            use_hybrid=request.use_hybrid and query_embedding is not None,
            min_score=0.0,
        )

        chunks = []
        arxiv_ids = []
        sources_set = set()

        for hit in search_results.get("hits", []):
            arxiv_id = hit.get("arxiv_id", "")

            chunks.append(
                {
                    "arxiv_id": arxiv_id,
                    "chunk_text": hit.get("chunk_text", hit.get("abstract", "")),
                }
            )

            if arxiv_id:
                arxiv_ids.append(arxiv_id)
                arxiv_id_clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
                sources_set.add(f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf")

        # End search span with metadata
        rag_tracer.end_search(search_span, chunks, arxiv_ids, search_results.get("total", 0))

    return chunks, list(sources_set), search_mode


@ask_nvidia.post("/ask_nvidia", response_model=AskResponse)
async def ask_nvidia_api(
    request: AskRequest,
    opensearch_client: OpenSearchDep,
    embeddings_service: EmbeddingsDep,
    nvidia_client: NvidiaDep,
    langfuse_tracer: LangfuseDep,
    cache_client: CacheDep,
):
    """Endpoint to ask questions to Nvidia with RAG + full tracing and caching."""

    rag_tracer = RAGTracer(langfuse_tracer)
    start_time = time.time()

    with rag_tracer.trace_request("api_user", request.query) as trace:
        try:
            # Check exact cache first
            if cache_client:
                try:
                    cached_response = await cache_client.find_cached_response(request)
                    if cached_response:
                        logger.info("Returning cached response for exact query match")
                        rag_tracer.end_request(trace, cached_response.answer, time.time() - start_time)
                        return cached_response
                except Exception as e:
                    logger.warning(f"Cache check failed, proceeding with normal flow: {e}")

            # Retrieve chunks and sources with tracing
            chunks, sources, search_mode = await _prepare_chunks_and_sources(
                request, opensearch_client, embeddings_service, rag_tracer, trace
            )

            if not chunks:
                answer = "Không tìm thấy tài liệu liên quan để trả lời câu hỏi của bạn."
                response = AskResponse(
                    query=request.query,
                    answer=answer,
                    sources=[],
                    chunks_used=0,
                    search_mode=search_mode,
                )
                rag_tracer.end_request(trace, answer, time.time() - start_time)
                return response
            with rag_tracer.trace_prompt_construction(trace, chunks) as prompt_span:
                from src.services.ollama.prompts import RAGPromptBuilder

                prompt_builder = RAGPromptBuilder()

                try:
                    prompt_data = prompt_builder.create_structured_prompt(request.query, chunks)
                    final_prompt = prompt_data["prompt"]
                except Exception:
                    final_prompt = prompt_builder.create_rag_prompt(request.query, chunks)

                rag_tracer.end_prompt(prompt_span, final_prompt)
            # Trace generation (Nvidia client handles prompt internally)
            with rag_tracer.trace_generation(trace, request.model, final_prompt) as gen_span:  # prompt không expose trực tiếp
                rag_result = await nvidia_client.generate_rag_answer(
                    query=request.query,
                    chunks=chunks,
                    model=request.model or None,
                    use_structured_output=True,
                )

                answer = rag_result.get("answer", "Không thể tạo câu trả lời.")
                rag_tracer.end_generation(gen_span, answer, request.model)

            # Prefer sources from Nvidia if available
            final_sources = rag_result.get("sources", sources)

            response = AskResponse(
                query=request.query,
                answer=answer,
                sources=final_sources,
                chunks_used=len(chunks),
                search_mode=search_mode,
            )

            rag_tracer.end_request(trace, answer, time.time() - start_time)

            # Cache the response
            if cache_client:
                try:
                    await cache_client.store_response(request, response)
                except Exception as e:
                    logger.warning(f"Failed to store response in cache: {e}")

                finally:
                    if hasattr(rag_tracer, "tracer") and rag_tracer.tracer:
                        rag_tracer.tracer.flush()  # hoặc langfuse_tracer.flush() nếu bạn có method này
            return response

        except Exception as e:
            logger.error(f"Error processing request: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Error processing request.")


@stream_nvidia.post("/stream_nvidia/stream")
async def stream_ask_nvidia(
    request: AskRequest,
    opensearch_client: OpenSearchDep,
    embeddings_service: EmbeddingsDep,
    nvidia_client: NvidiaDep,
    langfuse_tracer: LangfuseDep,
    cache_client: CacheDep,
) -> StreamingResponse:
    """Streamed endpoint with full tracing and caching support."""

    async def generate_stream():
        rag_tracer = RAGTracer(langfuse_tracer)
        start_time = time.time()

        with rag_tracer.trace_request("api_user", request.query) as trace:
            try:
                # Check cache first
                if cache_client:
                    try:
                        cached_response = await cache_client.find_cached_response(request)
                        if cached_response:
                            logger.info("Returning cached response for streaming query")

                            yield f"data: {json.dumps({'sources': cached_response.sources, 'chunks_used': cached_response.chunks_used, 'search_mode': cached_response.search_mode})}\n\n"

                            # Stream word by word
                            for word in cached_response.answer.split():
                                yield f"data: {json.dumps({'chunk': word + ' '})}\n\n"

                            yield f"data: {json.dumps({'answer': cached_response.answer, 'done': True})}\n\n"
                            rag_tracer.end_request(trace, cached_response.answer, time.time() - start_time)
                            return
                    except Exception as e:
                        logger.warning(f"Cache check failed: {e}")

                # Retrieve chunks with tracing
                chunks, sources, search_mode = await _prepare_chunks_and_sources(
                    request, opensearch_client, embeddings_service, rag_tracer, trace
                )

                if not chunks:
                    yield f"data: {json.dumps({'answer': 'Không tìm thấy tài liệu liên quan.', 'sources': [], 'done': True})}\n\n"
                    rag_tracer.end_request(trace, "Không tìm thấy tài liệu liên quan.", time.time() - start_time)
                    return

                # Send metadata
                yield f"data: {json.dumps({'sources': sources, 'chunks_used': len(chunks), 'search_mode': search_mode})}\n\n"

                with rag_tracer.trace_prompt_construction(trace, chunks) as prompt_span:
                    from src.services.ollama.prompts import RAGPromptBuilder

                    prompt_builder = RAGPromptBuilder()
                    logger.warning(f"request.query: {request.query}")
                    logger.warning(f"chunks: {chunks}")
                    final_prompt = prompt_builder.create_rag_prompt(request.query, chunks)
                    logger.warning(f"final_prompt: {type(final_prompt)}, length: {len(final_prompt)}")
                    rag_tracer.end_prompt(prompt_span, final_prompt)
                # Trace streaming generation
                with rag_tracer.trace_generation(trace, request.model, final_prompt) as gen_span:
                    full_response = ""
                    async for chunk in nvidia_client.generate_rag_answer_stream(
                        query=request.query,
                        chunks=chunks,
                        model=request.model,
                    ):
                        if chunk.get("response"):
                            text_chunk = chunk["response"]
                            full_response += text_chunk
                            yield f"data: {json.dumps({'chunk': text_chunk})}\n\n"

                        if chunk.get("done", False):
                            rag_tracer.end_generation(gen_span, full_response, request.model)
                            yield f"data: {json.dumps({'answer': full_response, 'done': True})}\n\n"
                            break

                rag_tracer.end_request(trace, full_response, time.time() - start_time)

                # Cache full response
                if cache_client and full_response:
                    try:
                        response_to_cache = AskResponse(
                            query=request.query,
                            answer=full_response,
                            sources=sources,
                            chunks_used=len(chunks),
                            search_mode=search_mode,
                        )
                        await cache_client.store_response(request, response_to_cache)
                    except Exception as e:
                        logger.warning(f"Failed to cache streaming response: {e}")

            except Exception as e:
                logger.error(f"Streaming error: {e}", exc_info=True)
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )

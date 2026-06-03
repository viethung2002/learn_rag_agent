import json
import logging
import time
from typing import Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from src.dependencies import CacheDep, EmbeddingsDep, EvaluationDep, LangfuseDep, OllamaDep, OpenSearchDep, Neo4jDep
from src.schemas.api.ask import AskRequest, AskResponse
from src.services.langfuse.tracer import RAGTracer
from src.services.neo4j import queries as neo4j_queries

logger = logging.getLogger(__name__)

# Two separate routers - one for regular ask, one for streaming
ask_router = APIRouter(tags=["ask"])
stream_router = APIRouter(tags=["stream"])


async def _prepare_chunks_and_sources(
    request: AskRequest,
    opensearch_client,
    embeddings_service,
    rag_tracer: RAGTracer,
    neo4j_client=None,
    trace=None,
) -> tuple[List[Dict], List[str], List[str]]:
    """Retrieve and prepare chunks for RAG with clean tracing."""

    # Handle embeddings for hybrid search
    query_embedding = None
    if request.use_hybrid:
        with rag_tracer.trace_embedding(trace, request.query) as embedding_span:
            try:
                query_embedding = await embeddings_service.embed_query(request.query)
                logger.info("Generated query embedding for hybrid search")
            except Exception as e:
                logger.warning(f"Failed to generate embeddings, falling back to BM25: {e}")
                if embedding_span:
                    rag_tracer.tracer.update_span(embedding_span, output={"success": False, "error": str(e)})

    # Search with tracing
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

        # Extract essential data for LLM
        chunks = []
        arxiv_ids = []
        sources_set = set()

        for hit in search_results.get("hits", []):
            arxiv_id = hit.get("arxiv_id", "")

            # Minimal chunk data for LLM
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
        # If Neo4j client provided, try to fetch graph facts and attach to chunks.
        if neo4j_client and arxiv_ids:
            try:
                query = neo4j_queries.build_papers_relations_query()
                rows = neo4j_client.execute_read(query, {"ids": arxiv_ids})
                facts_map = {}
                for r in rows:
                    simplified = neo4j_queries.simplify_relations_row(r)
                    facts_map[simplified.get("arxiv_id")] = simplified.get("relations", [])

                for c in chunks:
                    aid = c.get("arxiv_id")
                    if aid and aid in facts_map and facts_map[aid]:
                        c["graph_facts"] = facts_map[aid]
            except Exception as e:
                logger.warning(f"Neo4j query failed; continuing without graph facts: {e}")

        # End search span with essential metadata
        rag_tracer.end_search(search_span, chunks, arxiv_ids, search_results.get("total", 0))

    return chunks, list(sources_set), arxiv_ids


@ask_router.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    opensearch_client: OpenSearchDep,
    embeddings_service: EmbeddingsDep,
    ollama_client: OllamaDep,
    langfuse_tracer: LangfuseDep,
    neo4j_client: Neo4jDep,
    cache_client: CacheDep,
    evaluation_service: EvaluationDep,
) -> AskResponse:
    """Clean RAG endpoint with essential tracing and exact match caching."""

    rag_tracer = RAGTracer(langfuse_tracer)
    start_time = time.time()

    with rag_tracer.trace_request("api_user", request.query) as trace:
        try:
            # Check exact cache first
            cached_response = None
            if cache_client:
                try:
                    cached_response = await cache_client.find_cached_response(request)
                    if cached_response:
                        logger.info("Returning cached response for exact query match")
                        return cached_response
                except Exception as e:
                    logger.warning(f"Cache check failed, proceeding with normal flow: {e}")

            # Generate query embedding for hybrid search if needed
            query_embedding = None

            # Retrieve chunks
            chunks, sources, _ = await _prepare_chunks_and_sources(
                request, opensearch_client, embeddings_service, rag_tracer, neo4j_client, trace
            )

            if not chunks:
                response = AskResponse(
                    query=request.query,
                    answer="I couldn't find any relevant information in the papers to answer your question.",
                    sources=[],
                    chunks_used=0,
                    search_mode="bm25" if not request.use_hybrid else "hybrid",
                )
                rag_tracer.end_request(trace, response.answer, time.time() - start_time)
                return response

            # Build prompt
            with rag_tracer.trace_prompt_construction(trace, chunks) as prompt_span:
                from src.services.ollama.prompts import RAGPromptBuilder

                prompt_builder = RAGPromptBuilder()

                try:
                    prompt_data = prompt_builder.create_structured_prompt(request.query, chunks)
                    final_prompt = prompt_data["prompt"]
                except Exception:
                    final_prompt = prompt_builder.create_rag_prompt(request.query, chunks)

                rag_tracer.end_prompt(prompt_span, final_prompt)

            # Generate answer
            with rag_tracer.trace_generation(trace, request.model, final_prompt) as gen_span:
                rag_response = await ollama_client.generate_rag_answer(query=request.query, chunks=chunks, model=request.model)
                answer = rag_response.get("answer", "Unable to generate answer")
                rag_tracer.end_generation(gen_span, answer, request.model)

            # Prepare response
            evaluation = None
            if evaluation_service:
                evaluation = await evaluation_service.evaluate_answer(
                    query=request.query,
                    answer=answer,
                    contexts=[chunk.get("chunk_text", "") for chunk in chunks],
                    metadata={
                        "endpoint": "ask",
                        "model": request.model,
                        "search_mode": "hybrid" if request.use_hybrid else "bm25",
                    },
                )

            response = AskResponse(
                query=request.query,
                answer=answer,
                sources=sources,
                chunks_used=len(chunks),
                search_mode="bm25" if not request.use_hybrid else "hybrid",
                evaluation=evaluation,
            )

            rag_tracer.end_request(trace, answer, time.time() - start_time)

            # Store response in exact match cache
            if cache_client:
                try:
                    await cache_client.store_response(request, response)
                except Exception as e:
                    logger.warning(f"Failed to store response in cache: {e}")

            return response

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@stream_router.post("/stream")
async def ask_question_stream(
    request: AskRequest,
    opensearch_client: OpenSearchDep,
    embeddings_service: EmbeddingsDep,
    ollama_client: OllamaDep,
    langfuse_tracer: LangfuseDep,
    neo4j_client: Neo4jDep,
    cache_client: CacheDep,
) -> StreamingResponse:
    """Clean streaming RAG endpoint."""

    async def generate_stream():
        rag_tracer = RAGTracer(langfuse_tracer)
        start_time = time.time()

        with rag_tracer.trace_request("api_user", request.query) as trace:
            try:
                # Check exact cache first
                if cache_client:
                    try:
                        cached_response = await cache_client.find_cached_response(request)
                        if cached_response:
                            logger.info("Returning cached response for exact streaming query match")

                            # Send metadata first (same format as non-cached)
                            metadata_response = {
                                "sources": cached_response.sources,
                                "chunks_used": cached_response.chunks_used,
                                "search_mode": cached_response.search_mode,
                            }
                            yield f"data: {json.dumps(metadata_response)}\n\n"

                            # Stream the cached response in chunks
                            for chunk in cached_response.answer.split():
                                yield f"data: {json.dumps({'chunk': chunk + ' '})}\n\n"

                            # Send completion signal with just the final answer
                            yield f"data: {json.dumps({'answer': cached_response.answer, 'done': True})}\n\n"
                            return
                    except Exception as e:
                        logger.warning(f"Cache check failed, proceeding with normal flow: {e}")

                # Retrieve chunks
                chunks, sources, _ = await _prepare_chunks_and_sources(
                    request, opensearch_client, embeddings_service, rag_tracer, neo4j_client, trace
                )

                if not chunks:
                    yield f"data: {json.dumps({'answer': 'No relevant information found.', 'sources': [], 'done': True})}\n\n"
                    return

                # Send metadata first
                search_mode = "bm25" if not request.use_hybrid else "hybrid"
                metadata_response = {"sources": sources, "chunks_used": len(chunks), "search_mode": search_mode}
                yield f"data: {json.dumps(metadata_response)}\n\n"

                # Build prompt
                with rag_tracer.trace_prompt_construction(trace, chunks) as prompt_span:
                    from src.services.ollama.prompts import RAGPromptBuilder

                    prompt_builder = RAGPromptBuilder()
                    final_prompt = prompt_builder.create_rag_prompt(request.query, chunks)
                    rag_tracer.end_prompt(prompt_span, final_prompt)

                # Stream generation
                with rag_tracer.trace_generation(trace, request.model, final_prompt) as gen_span:
                    full_response = ""
                    async for chunk in ollama_client.generate_rag_answer_stream(
                        query=request.query, chunks=chunks, model=request.model
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

                # Store response in exact match cache
                if cache_client and full_response:
                    try:
                        search_mode = "bm25" if not request.use_hybrid else "hybrid"
                        response_to_cache = AskResponse(
                            query=request.query,
                            answer=full_response,
                            sources=sources,
                            chunks_used=len(chunks),
                            search_mode=search_mode,
                        )
                        await cache_client.store_response(request, response_to_cache)
                    except Exception as e:
                        logger.warning(f"Failed to store streaming response in cache: {e}")

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate_stream(), media_type="text/plain", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

import asyncio
import logging
import math
import re
import time
from typing import Any, Dict, List, Literal, Optional

from fastmcp import FastMCP
from langchain_core.documents import Document
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

from src.config import get_settings
from src.services.embeddings.factory import make_embeddings_client
from src.services.neo4j import queries as neo4j_queries
from src.services.neo4j.factory import make_neo4j_client
from src.services.nvidia.factory import make_nvidia_client
from src.services.opensearch.factory import make_opensearch_client

mcp = FastMCP("arxiv-tools")
logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 4
DEFAULT_GRAPH_TIMEOUT_SECONDS = 5.0
DEFAULT_SEARCH_TIMEOUT_SECONDS = 5.0
ROUTING_CACHE_SIZE = 256

_GRAPH_CITATION_MARKERS = (
    "shared citations",
    "citations do",
    "references do",
    "cited by both",
    "share citations",
    "share references",
    "share the same citations",
    "common citations",
    "common references",
    "citation overlap",
    "reference overlap",
    "citation network",
    "citation graph",
    "citation path",
    "graph citation",
    "ancestor citation",
    "descendant citation",
    "paper graph",
)
_SEARCH_MARKERS = (
    "similar",
    "semantically similar",
    "nearest neighbors",
    "nearest neighbour",
    "related papers",
    "find papers",
    "search papers",
    "retrieve papers",
    "papers about",
    "survey",
    "overview",
)
_BOTH_MARKERS = (
    "and also",
    "along with",
    "together with",
    "both",
)
_routing_cache: Dict[str, "ToolRoutingDecision"] = {}


class ToolRoutingDecision(BaseModel):
    action: Literal["graph", "search", "both"] = Field(
        description="Which retrieval tool(s) should be executed."
    )
    query: str = Field(description="Normalized query text to pass to the tool(s).")
    graph_reason: str = Field(default="", description="Why graph retrieval is or is not needed.")
    search_reason: str = Field(default="", description="Why search retrieval is or is not needed.")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    graph_enabled: bool = Field(default=False)
    search_enabled: bool = Field(default=True)


class GraphCitationIntent(BaseModel):
    should_use_graph: bool = Field(
        description="Whether the question is best answered by graph traversal in Neo4j."
    )
    paper_titles: List[str] = Field(
        default_factory=list,
        description="Paper titles explicitly or implicitly mentioned in the query.",
    )
    reasoning: str = Field(default="")


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", (query or "")).strip()


def _cache_get(key: str) -> Optional[ToolRoutingDecision]:
    decision = _routing_cache.get(key)
    if decision is None:
        return None
    _routing_cache.pop(key, None)
    _routing_cache[key] = decision
    return decision.model_copy(deep=True)


def _cache_put(key: str, decision: ToolRoutingDecision) -> None:
    _routing_cache[key] = decision.model_copy(deep=True)
    while len(_routing_cache) > ROUTING_CACHE_SIZE:
        oldest = next(iter(_routing_cache))
        _routing_cache.pop(oldest, None)


def _extract_quoted_titles(query: str) -> List[str]:
    matches = re.findall(r"""['"]([^'"]+)['"]""", query or "")
    titles: List[str] = []
    for match in matches:
        cleaned = re.sub(r"\s+", " ", match).strip()
        if cleaned and cleaned not in titles:
            titles.append(cleaned)
    return titles


def _looks_like_shared_citation_query(query: str) -> bool:
    lowered = (query or "").lower()
    has_citation_marker = any(marker in lowered for marker in _GRAPH_CITATION_MARKERS)
    has_pair_marker = "both" in lowered or " and " in lowered
    return has_citation_marker and has_pair_marker


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip().lower()


def _tokenize_title(value: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", _normalize_text(value))
        if len(token) >= 3
    ]


def _token_overlap_ratio(left: str, right: str) -> float:
    left_tokens = set(_tokenize_title(left))
    right_tokens = set(_tokenize_title(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return -1.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return -1.0
    return dot / (norm_a * norm_b)


def _build_pdf_url(arxiv_id: Optional[str]) -> str:
    if not arxiv_id:
        return ""
    clean_id = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
    return f"https://arxiv.org/pdf/{clean_id}.pdf"


def _heuristic_route(query: str) -> ToolRoutingDecision:
    normalized_query = _normalize_query(query)
    lowered = normalized_query.lower()
    quoted_titles = _extract_quoted_titles(normalized_query)

    graph_hit = any(marker in lowered for marker in _GRAPH_CITATION_MARKERS)
    search_hit = any(marker in lowered for marker in _SEARCH_MARKERS)
    both_hint = any(marker in lowered for marker in _BOTH_MARKERS)

    if len(quoted_titles) >= 2 and _looks_like_shared_citation_query(normalized_query):
        graph_hit = True

    if graph_hit and (search_hit or both_hint):
        return ToolRoutingDecision(
            action="both",
            query=normalized_query,
            graph_reason="Query asks for citation-graph relations.",
            search_reason="Query also asks for semantic or topical retrieval.",
            confidence=0.9,
            graph_enabled=True,
            search_enabled=True,
        )

    if graph_hit:
        return ToolRoutingDecision(
            action="graph",
            query=normalized_query,
            graph_reason="Query is graph-native and best served by Neo4j traversal.",
            search_reason="Semantic retrieval is not required by the wording.",
            confidence=0.82,
            graph_enabled=True,
            search_enabled=False,
        )

    return ToolRoutingDecision(
        action="search",
        query=normalized_query,
        graph_reason="No graph-specific citation traversal signal detected.",
        search_reason="Defaulting to OpenSearch for paper retrieval.",
        confidence=0.7 if search_hit else 0.6,
        graph_enabled=False,
        search_enabled=True,
    )


async def _route_with_llm(query: str, heuristic: ToolRoutingDecision) -> ToolRoutingDecision:
    normalized_query = _normalize_query(query)
    cache_key = normalized_query.lower()
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    settings = get_settings()
    if not settings.nvidia_api_key:
        _cache_put(cache_key, heuristic)
        return heuristic

    try:
        nvidia_client = make_nvidia_client()
        llm = nvidia_client.get_langchain_model(
            model=settings.nvidia_model,
            temperature=0.0,
        )
        structured_llm = llm.with_structured_output(ToolRoutingDecision)
        prompt = (
            "You are a routing model for arXiv retrieval tools.\n"
            "Choose one action from: graph, search, both.\n"
            "Tool graph_citation_retrieve: use for shared citations, citation paths, citation overlap, common references, paper graph questions.\n"
            "Tool retrieve_papers: use for semantic paper search, related papers, similar work, surveys, topical retrieval.\n"
            "Choose both only when the user clearly needs both graph relations and semantic retrieval.\n"
            "Prefer graph for: citation network, ancestors, descendants, paths, shared references.\n"
            "Prefer search for: similar papers, semantically similar, nearest neighbors, papers about a topic.\n"
            "Return only structured output.\n\n"
            f"User query: {normalized_query}\n"
            f"Heuristic suggestion: {heuristic.model_dump_json()}"
        )
        decision = await structured_llm.ainvoke(prompt)
        decision.query = normalized_query
        _cache_put(cache_key, decision)
        return decision
    except Exception as exc:
        logger.warning("LLM routing failed, falling back to heuristics: %s", exc)
        _cache_put(cache_key, heuristic)
        return heuristic


async def _extract_graph_citation_intent(query: str) -> GraphCitationIntent:
    quoted_titles = _extract_quoted_titles(query)
    if len(quoted_titles) >= 2 and _looks_like_shared_citation_query(query):
        return GraphCitationIntent(
            should_use_graph=True,
            paper_titles=quoted_titles[:2],
            reasoning="Detected shared-citation query with two quoted titles.",
        )

    settings = get_settings()
    if not settings.nvidia_api_key:
        return GraphCitationIntent(
            should_use_graph=False,
            paper_titles=quoted_titles[:2],
            reasoning="No LLM configured for graph intent extraction.",
        )

    try:
        nvidia_client = make_nvidia_client()
        llm = nvidia_client.get_langchain_model(
            model=settings.nvidia_model,
            temperature=0.0,
        )
        structured_llm = llm.with_structured_output(GraphCitationIntent)
        prompt = (
            "Extract paper titles and decide whether the question should be answered "
            "by graph traversal in Neo4j.\n"
            "Use graph traversal for shared citations, reference overlap, citation paths, "
            "citation network, ancestors, descendants, or common bibliography.\n\n"
            f"Question: {query}"
        )
        result = await structured_llm.ainvoke(prompt)
        deduped: List[str] = []
        for title in result.paper_titles:
            cleaned = re.sub(r"\s+", " ", title).strip()
            if cleaned and cleaned not in deduped:
                deduped.append(cleaned)
        result.paper_titles = deduped[:2]
        return result
    except Exception as exc:
        logger.warning("Graph citation intent extraction failed: %s", exc)
        return GraphCitationIntent(
            should_use_graph=False,
            paper_titles=quoted_titles[:2],
            reasoning="Fallback after LLM extraction failure.",
        )


async def _resolve_paper_by_title_via_graph(
    title: str,
    neo4j_client: Any,
    embeddings_client: Any,
) -> Optional[Dict[str, Any]]:
    title_query = _normalize_text(title)
    title_tokens = _tokenize_title(title)
    loop = asyncio.get_running_loop()
    rows = await loop.run_in_executor(
        None,
        lambda: neo4j_client.execute_read(
            neo4j_queries.build_paper_title_candidates_query(),
            {"title_query": title_query, "title_tokens": title_tokens},
        ),
    )
    if not rows:
        return None

    try:
        query_embedding = await embeddings_client.embed_query(title)
        candidate_titles = [row.get("title", "") for row in rows]
        candidate_embeddings = await embeddings_client.embed_passages(candidate_titles)
    except Exception as exc:
        logger.warning("Embedding rerank failed for title '%s': %s", title, exc)
        query_embedding = []
        candidate_embeddings = [[] for _ in rows]

    best_row: Optional[Dict[str, Any]] = None
    best_score = -1.0
    normalized_target = _normalize_text(title)
    for row, candidate_embedding in zip(rows, candidate_embeddings):
        candidate_title = row.get("title", "")
        normalized_candidate = _normalize_text(candidate_title)
        lexical_bonus = 0.15 if normalized_candidate == normalized_target else 0.0
        overlap_score = _token_overlap_ratio(title, candidate_title)
        embedding_score = _cosine_similarity(query_embedding, candidate_embedding)
        score = embedding_score + overlap_score + lexical_bonus
        if score > best_score:
            best_score = score
            best_row = row

    if not best_row:
        return None

    match_overlap = _token_overlap_ratio(title, best_row.get("title", ""))
    exact_match = _normalize_text(best_row.get("title", "")) == normalized_target
    if not exact_match and match_overlap < 0.45 and best_score < 1.10:
        return None

    return {
        "arxiv_id": best_row.get("arxiv_id", ""),
        "title": best_row.get("title", title),
        "abstract": best_row.get("abstract", ""),
        "score": best_score,
        "token_overlap": match_overlap,
        "url": _build_pdf_url(best_row.get("arxiv_id")),
        "resolver": "neo4j+embedding",
    }


async def call_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    use_hybrid: bool = True,
) -> List[Document]:
    opensearch_client = make_opensearch_client()
    embeddings_client = make_embeddings_client()

    logger.info("retrieve_papers.start query=%s top_k=%s", query[:100], top_k)
    query_embedding = await embeddings_client.embed_query(query)

    loop = asyncio.get_running_loop()
    search_results = await loop.run_in_executor(
        None,
        lambda: opensearch_client.search_unified(
            query=query,
            query_embedding=query_embedding,
            size=top_k,
            use_hybrid=use_hybrid,
        ),
    )

    documents: List[Document] = []
    hits = search_results.get("hits", [])
    seen_ids: set[str] = set()
    for hit in hits:
        arxiv_id = hit.get("arxiv_id", "")
        if arxiv_id and arxiv_id in seen_ids:
            continue
        if arxiv_id:
            seen_ids.add(arxiv_id)
        documents.append(
            Document(
                page_content=hit["chunk_text"],
                metadata={
                    "arxiv_id": arxiv_id,
                    "title": hit.get("title", ""),
                    "authors": hit.get("authors", ""),
                    "score": hit.get("score", 0.0),
                    "source": _build_pdf_url(arxiv_id),
                    "section": hit.get("section_name", ""),
                    "search_mode": "hybrid" if use_hybrid else "bm25",
                    "top_k": top_k,
                    "provenance": "opensearch",
                },
            )
        )

    logger.info("retrieve_papers.success count=%s", len(documents))
    return documents


async def call_graph(query: str) -> List[Document]:
    neo4j_client = make_neo4j_client()
    embeddings_client = make_embeddings_client()

    intent = await _extract_graph_citation_intent(query)
    if not intent.should_use_graph or len(intent.paper_titles) < 2:
        logger.info("graph_citation_retrieve.skipped reason=%s", intent.reasoning)
        return []

    paper_a = await _resolve_paper_by_title_via_graph(
        intent.paper_titles[0],
        neo4j_client,
        embeddings_client,
    )
    paper_b = await _resolve_paper_by_title_via_graph(
        intent.paper_titles[1],
        neo4j_client,
        embeddings_client,
    )
    if not paper_a or not paper_b:
        logger.info("graph_citation_retrieve.unresolved titles=%s", intent.paper_titles)
        return []

    loop = asyncio.get_running_loop()
    rows = await loop.run_in_executor(
        None,
        lambda: neo4j_client.execute_read(
            neo4j_queries.build_shared_citations_query(),
            {"a": paper_a["arxiv_id"], "b": paper_b["arxiv_id"]},
        ),
    )

    documents: List[Document] = []
    for row in rows:
        cited_title = row.get("title") or "Untitled reference"
        cited_arxiv_id = row.get("arxiv_id") or ""
        documents.append(
            Document(
                page_content=(
                    f"Shared citation between '{paper_a['title']}' and '{paper_b['title']}': "
                    f"{cited_title}" + (f" (arXiv:{cited_arxiv_id})" if cited_arxiv_id else "")
                ),
                metadata={
                    "arxiv_id": cited_arxiv_id,
                    "title": cited_title,
                    "authors": [],
                    "score": 1.0,
                    "source": _build_pdf_url(cited_arxiv_id),
                    "section": "shared_citation",
                    "search_mode": "neo4j_graph",
                    "top_k": len(rows),
                    "labels": row.get("labels", []),
                    "source_papers": [paper_a["arxiv_id"], paper_b["arxiv_id"]],
                    "provenance": "neo4j",
                    "graph_reasoning": intent.reasoning,
                },
            )
        )

    logger.info(
        "graph_citation_retrieve.success source_a=%s source_b=%s count=%s",
        paper_a["arxiv_id"],
        paper_b["arxiv_id"],
        len(documents),
    )
    return documents


def _document_to_item(document: Document, tool_name: str) -> Dict[str, Any]:
    metadata = dict(document.metadata or {})
    return {
        "page_content": document.page_content,
        "metadata": metadata,
        "source": metadata.get("provenance") or tool_name,
        "tool_name": tool_name,
        "score": metadata.get("score", 0.0),
        "arxiv_id": metadata.get("arxiv_id", ""),
        "title": metadata.get("title", ""),
    }


def _dedupe_key(item: Dict[str, Any]) -> str:
    if item.get("arxiv_id"):
        return f"arxiv:{item['arxiv_id']}"
    title = _normalize_text(item.get("title", ""))
    if title:
        return f"title:{title}"
    return f"content:{_normalize_text(item.get('page_content', ''))[:120]}"


def merge_routed_results(results_by_tool: Dict[str, List[Document]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for tool_name, documents in results_by_tool.items():
        for document in documents:
            item = _document_to_item(document, tool_name)
            item_key = _dedupe_key(item)
            existing = merged.get(item_key)
            if existing is None:
                item["sources"] = [item["source"]]
                item["tool_names"] = [tool_name]
                merged[item_key] = item
                continue

            existing["score"] = max(existing.get("score", 0.0), item.get("score", 0.0))
            existing["tool_names"] = sorted(set(existing.get("tool_names", []) + [tool_name]))
            existing["sources"] = sorted(set(existing.get("sources", []) + [item["source"]]))
            if len(item.get("page_content", "")) > len(existing.get("page_content", "")):
                existing["page_content"] = item["page_content"]
            existing_metadata = existing.setdefault("metadata", {})
            incoming_metadata = item.get("metadata", {})
            for field in ("section", "search_mode", "source_papers", "labels", "graph_reasoning"):
                if field not in existing_metadata and field in incoming_metadata:
                    existing_metadata[field] = incoming_metadata[field]
            if not existing_metadata.get("source") and incoming_metadata.get("source"):
                existing_metadata["source"] = incoming_metadata["source"]

    return sorted(
        merged.values(),
        key=lambda item: item.get("score", 0.0),
        reverse=True,
    )


async def _execute_with_timeout(
    tool_name: str,
    coroutine: Any,
    timeout_seconds: float,
) -> Dict[str, Any]:
    start = time.perf_counter()
    try:
        documents = await asyncio.wait_for(coroutine, timeout=timeout_seconds)
        return {
            "tool_name": tool_name,
            "documents": documents,
            "duration_seconds": round(time.perf_counter() - start, 4),
            "warning": None,
        }
    except Exception as exc:
        logger.warning("tool execution failed tool=%s error=%s", tool_name, exc)
        return {
            "tool_name": tool_name,
            "documents": [],
            "duration_seconds": round(time.perf_counter() - start, 4),
            "warning": str(exc),
        }


async def decide_tools_with_llm(
    query: str,
    context: Optional[Dict[str, Any]] = None,
) -> ToolRoutingDecision:
    del context
    heuristic = _heuristic_route(query)
    decision = await _route_with_llm(query, heuristic)

    if decision.action == "graph" and not decision.graph_enabled:
        return heuristic
    if decision.action == "search" and not decision.search_enabled:
        return heuristic
    if decision.action == "both" and (not decision.graph_enabled or not decision.search_enabled):
        return heuristic
    return decision


async def execute_routing(
    decision: ToolRoutingDecision,
    graph_timeout_seconds: float = DEFAULT_GRAPH_TIMEOUT_SECONDS,
    search_timeout_seconds: float = DEFAULT_SEARCH_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    task_specs: List[tuple[str, Any, float]] = []
    if decision.action in ("graph", "both"):
        task_specs.append(("graph_citation_retrieve", call_graph(decision.query), graph_timeout_seconds))
    if decision.action in ("search", "both"):
        task_specs.append(("retrieve_papers", call_search(decision.query), search_timeout_seconds))

    executions = await asyncio.gather(
        *[
            _execute_with_timeout(tool_name, coroutine, timeout_seconds)
            for tool_name, coroutine, timeout_seconds in task_specs
        ]
    )

    results_by_tool = {
        execution["tool_name"]: execution["documents"]
        for execution in executions
    }
    merged_results = merge_routed_results(results_by_tool)
    warnings = [execution["warning"] for execution in executions if execution["warning"]]

    return {
        "action": decision.action,
        "query": decision.query,
        "results": merged_results,
        "metadata": {
            "router_confidence": decision.confidence,
            "graph_reason": decision.graph_reason,
            "search_reason": decision.search_reason,
            "which_tool": [execution["tool_name"] for execution in executions],
            "latencies": {
                execution["tool_name"]: execution["duration_seconds"]
                for execution in executions
            },
            "warnings": warnings,
            "partial_results": bool(warnings),
        },
    }


@mcp.tool()
async def retrieve_papers(query: str) -> List[Document]:
    """Search and return relevant arXiv research papers."""
    return await call_search(query=query)


@mcp.tool()
async def graph_citation_retrieve(query: str) -> List[Document]:
    """Resolve graph-native citation queries against Neo4j."""
    return await call_graph(query=query)


@mcp.tool()
async def route_query_via_llm(
    query: str,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Route a query to graph retrieval, semantic search, or both."""
    decision = await decide_tools_with_llm(query=query, context=context)
    return await execute_routing(decision)


@mcp.tool()
async def web_search(query: str) -> str:
    """Perform a web search to gather current information."""
    search_tool = TavilySearch(
        max_results=2,
        tavily_api_key="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",
    )
    results = await search_tool.ainvoke({"query": query})

    if not results or "results" not in results:
        return "No search results found."

    summaries = []
    for result in results["results"]:
        title = result.get("title", "No title")
        content = result.get("content", "")
        url = result.get("url", "")
        summaries.append(f"- {title}\n  {content}\n  Source: {url}")

    return "\n".join(summaries)


if __name__ == "__main__":
    opensearch_client = make_opensearch_client()
    print("Health check:", opensearch_client.health_check())
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8100,
        path="/mcp",
    )

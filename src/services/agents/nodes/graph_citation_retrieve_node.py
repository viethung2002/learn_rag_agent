import logging
import math
import re
from typing import Any, Dict, List, Optional

from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from src.services.neo4j import queries as neo4j_queries

from ..context import Context
from ..state import AgentState
from .utils import get_latest_query

logger = logging.getLogger(__name__)

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
)


class GraphCitationIntent(BaseModel):
    """Structured intent for graph-first citation retrieval."""

    should_use_graph: bool = Field(
        description="Whether the question is best answered by graph traversal in Neo4j."
    )
    paper_titles: List[str] = Field(
        default_factory=list,
        description="Paper titles explicitly or implicitly mentioned in the user query.",
    )
    reasoning: str = Field(default="", description="Short explanation for the extraction.")


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


async def _extract_graph_citation_intent(query: str, runtime: Runtime[Context]) -> GraphCitationIntent:
    quoted_titles = _extract_quoted_titles(query)
    if len(quoted_titles) >= 2 and _looks_like_shared_citation_query(query):
        return GraphCitationIntent(
            should_use_graph=True,
            paper_titles=quoted_titles[:2],
            reasoning="Detected shared-citation query with two quoted paper titles.",
        )

    llm = runtime.context.nvidia_client.get_langchain_model(
        model=runtime.context.model_name,
        temperature=0.0,
    )
    structured_llm = llm.with_structured_output(GraphCitationIntent)
    prompt = (
        "Extract paper titles and decide whether the question should be answered by graph traversal in Neo4j.\n"
        "Use graph traversal when the user asks about shared citations, references, citation overlap, or common bibliography.\n\n"
        f"Question: {query}"
    )
    try:
        result = await structured_llm.ainvoke(prompt)
        if result.paper_titles:
            deduped: List[str] = []
            for title in result.paper_titles:
                cleaned = re.sub(r"\s+", " ", title).strip()
                if cleaned and cleaned not in deduped:
                    deduped.append(cleaned)
            result.paper_titles = deduped[:2]
        return result
    except Exception as e:
        logger.warning("Graph citation intent extraction failed: %s", e)
        return GraphCitationIntent(
            should_use_graph=False,
            paper_titles=quoted_titles[:2],
            reasoning="Fallback after LLM extraction failure.",
        )


async def _resolve_paper_by_title_via_graph(
    title: str,
    runtime: Runtime[Context],
) -> Optional[Dict[str, Any]]:
    neo4j_client = runtime.context.neo4j_client
    embeddings_client = runtime.context.embeddings_client
    if not neo4j_client:
        return None

    title_query = _normalize_text(title)
    title_tokens = _tokenize_title(title)
    rows = neo4j_client.execute_read(
        neo4j_queries.build_paper_title_candidates_query(),
        {
            "title_query": title_query,
            "title_tokens": title_tokens,
        },
    )
    if not rows:
        logger.info("Graph title lookup found no candidates for '%s'", title)
        return None

    try:
        query_embedding = await embeddings_client.embed_query(title)
        candidate_titles = [row.get("title", "") for row in rows]
        candidate_embeddings = await embeddings_client.embed_passages(candidate_titles)
    except Exception as e:
        logger.warning("Embedding rerank failed for title '%s': %s", title, e)
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
        logger.info(
            "Graph title lookup rejected low-confidence match for '%s': candidate='%s' score=%.3f overlap=%.3f",
            title,
            best_row.get("title", ""),
            best_score,
            match_overlap,
        )
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


async def ainvoke_graph_citation_retrieve_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, Any]:
    """Resolve paper titles and fetch shared citations directly from Neo4j.

    This node is intended for graph-native citation questions where OpenSearch
    chunk retrieval is a poor fit, for example "Which citations do paper A and
    paper B share?".
    """

    query = get_latest_query(state["messages"])
    neo4j_client = runtime.context.neo4j_client
    if not neo4j_client:
        logger.info("Graph citation retrieve skipped: Neo4j client unavailable")
        return {
            "metadata": {
                "graph_query_applicable": False,
                "graph_query_reason": "neo4j_unavailable",
            }
        }

    intent = await _extract_graph_citation_intent(query, runtime)
    if not intent.should_use_graph or len(intent.paper_titles) < 2:
        logger.info("Graph citation retrieve skipped: query not suitable for graph-first retrieval")
        return {
            "metadata": {
                "graph_query_applicable": False,
                "graph_query_reason": intent.reasoning or "not_graph_citation_query",
            }
        }

    paper_a = await _resolve_paper_by_title_via_graph(intent.paper_titles[0], runtime)
    paper_b = await _resolve_paper_by_title_via_graph(intent.paper_titles[1], runtime)
    if not paper_a or not paper_b:
        logger.info("Graph citation retrieve failed to resolve both titles: %s", intent.paper_titles)
        return {
            "metadata": {
                "graph_query_applicable": False,
                "graph_query_reason": "title_resolution_failed",
                "graph_query_titles": intent.paper_titles,
                "neo4j_attempted": True,
            }
        }

    rows = neo4j_client.execute_read(
        neo4j_queries.build_shared_citations_query(),
        {"a": paper_a["arxiv_id"], "b": paper_b["arxiv_id"]},
    )

    shared_docs: List[Dict[str, Any]] = []
    for row in rows:
        cited_title = row.get("title") or "Untitled reference"
        cited_arxiv_id = row.get("arxiv_id") or ""
        shared_docs.append(
            {
                "page_content": (
                    f"Shared citation between '{paper_a['title']}' and '{paper_b['title']}': "
                    f"{cited_title}" + (f" (arXiv:{cited_arxiv_id})" if cited_arxiv_id else "")
                ),
                "metadata": {
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
                },
            }
        )

    if not shared_docs:
        shared_docs.append(
            {
                "page_content": (
                    f"No shared citations found between '{paper_a['title']}' and '{paper_b['title']}'."
                ),
                "metadata": {
                    "arxiv_id": "",
                    "title": "No shared citations found",
                    "authors": [],
                    "score": 0.0,
                    "source": "",
                    "section": "shared_citation",
                    "search_mode": "neo4j_graph",
                    "top_k": 0,
                    "labels": [],
                    "source_papers": [paper_a["arxiv_id"], paper_b["arxiv_id"]],
                },
            }
        )

    resolved_same_paper = bool(paper_a.get("arxiv_id")) and paper_a.get("arxiv_id") == paper_b.get("arxiv_id")

    logger.info(
        "Graph citation retrieve resolved papers %s and %s with %s shared citation(s)",
        paper_a["arxiv_id"],
        paper_b["arxiv_id"],
        len(shared_docs),
    )

    return {
        "retrieved_docs": shared_docs,
        "relevant_sources": [paper_a, paper_b],
        "metadata": {
            "graph_query_applicable": True,
            "graph_query_reason": intent.reasoning,
            "neo4j_attempted": True,
            "used_neo4j": True,
            "graph_enriched_docs": len(shared_docs),
            "graph_enriched_arxiv_ids": [paper_a["arxiv_id"], paper_b["arxiv_id"]],
            "graph_query_titles": intent.paper_titles,
            "graph_query_type": "shared_citations",
            "graph_query_rows": len(rows),
            "graph_query_satisfied": True,
            "resolved_same_paper": resolved_same_paper,
        },
    }


def route_after_graph_citation_retrieve(state: AgentState) -> str:
    """Route graph-first citation retrieval to rerank or fallback retrieval."""

    metadata = state.get("metadata", {}) or {}
    if metadata.get("graph_query_applicable"):
        logger.info("Graph citation retrieve applicable -> rerank graph results")
        return "rerank"

    logger.info("Graph citation retrieve not applicable -> fallback to standard retrieve")
    return "retrieve"

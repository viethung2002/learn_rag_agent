import logging
import re
from typing import Any, Dict, List, Optional

from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from ..context import Context
from ..state import AgentState
from .utils import get_latest_query

logger = logging.getLogger(__name__)

_ARXIV_ID_PATTERN = re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", re.IGNORECASE)
_GRAPH_KEYWORDS = (
    "neo4j",
    "cypher",
    "graph query",
    "graph truy",
    "truy vấn neo4j",
    "query tren neo4j",
    "query trên neo4j",
    "cau lenh neo4j",
    "câu lệnh neo4j",
    "graph database",
)
_GRAPH_FACT_KEYWORDS = (
    "title",
    "abstract",
    "author",
    "authors",
    "reference",
    "references",
    "citation",
    "citations",
    "cited",
    "arxiv id",
    "paper with arxiv id",
    "wrote",
)
_TITLE_TO_ARXIV_KEYWORDS = (
    "arxiv id",
    "arxiv identifier",
    "identifier",
    "article id",
    "paper id",
)


class GraphRetrieveIntent(BaseModel):
    """Structured graph retrieval intent determined from the user query."""

    should_use_graph: bool = Field(
        description="Whether the question is better served by Neo4j graph retrieval/query generation."
    )
    query_kind: str = Field(
        default="unknown",
        description="High-level query family such as paper_lookup, citation, author_lookup, or cypher_generation.",
    )
    arxiv_ids: List[str] = Field(
        default_factory=list,
        description="Explicit arXiv IDs found in the question.",
    )
    paper_titles: List[str] = Field(
        default_factory=list,
        description="Quoted paper titles found in the question.",
    )
    reasoning: str = Field(default="", description="Short heuristic explanation.")


def _extract_quoted_titles(query: str) -> List[str]:
    matches = re.findall(r"""['"]([^'"]+)['"]""", query or "")
    titles: List[str] = []
    for match in matches:
        cleaned = re.sub(r"\s+", " ", match).strip()
        if cleaned and cleaned not in titles:
            titles.append(cleaned)
    return titles


def _extract_arxiv_ids(query: str) -> List[str]:
    ids: List[str] = []
    for match in _ARXIV_ID_PATTERN.findall(query or ""):
        normalized = match.strip()
        if normalized and normalized not in ids:
            ids.append(normalized)
    return ids


def _classify_graph_query(query: str) -> GraphRetrieveIntent:
    lowered = (query or "").lower()
    arxiv_ids = _extract_arxiv_ids(query)
    paper_titles = _extract_quoted_titles(query)

    if any(keyword in lowered for keyword in _GRAPH_KEYWORDS):
        return GraphRetrieveIntent(
            should_use_graph=True,
            query_kind="cypher_generation",
            arxiv_ids=arxiv_ids,
            paper_titles=paper_titles,
            reasoning="explicit_neo4j_or_cypher_request",
        )

    if arxiv_ids and any(keyword in lowered for keyword in _GRAPH_FACT_KEYWORDS):
        return GraphRetrieveIntent(
            should_use_graph=True,
            query_kind="paper_lookup",
            arxiv_ids=arxiv_ids,
            paper_titles=paper_titles,
            reasoning="arxiv_id_lookup_query",
        )

    if paper_titles and any(keyword in lowered for keyword in _TITLE_TO_ARXIV_KEYWORDS):
        return GraphRetrieveIntent(
            should_use_graph=True,
            query_kind="title_to_arxiv_lookup",
            arxiv_ids=arxiv_ids,
            paper_titles=paper_titles,
            reasoning="paper_title_to_arxiv_lookup_query",
        )

    if len(paper_titles) >= 2 and any(
        marker in lowered
        for marker in ("shared citations", "shared references", "cited by both", "common citations", "common references")
    ):
        return GraphRetrieveIntent(
            should_use_graph=True,
            query_kind="shared_citations",
            arxiv_ids=arxiv_ids,
            paper_titles=paper_titles[:2],
            reasoning="shared_citation_query",
        )

    if paper_titles and any(keyword in lowered for keyword in ("author", "authors", "title", "abstract", "citation", "reference")):
        return GraphRetrieveIntent(
            should_use_graph=True,
            query_kind="title_lookup",
            arxiv_ids=arxiv_ids,
            paper_titles=paper_titles,
            reasoning="paper_title_graph_lookup",
        )

    return GraphRetrieveIntent(
        should_use_graph=False,
        query_kind="unknown",
        arxiv_ids=arxiv_ids,
        paper_titles=paper_titles,
        reasoning="not_graph_query",
    )


def _build_pdf_url(arxiv_id: Optional[str]) -> str:
    if not arxiv_id:
        return ""
    clean_id = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
    return f"https://arxiv.org/pdf/{clean_id}.pdf"


def _build_schema_summary() -> str:
    return (
        "Neo4j graph schema available in this project:\n"
        "- Node :Paper properties: arxiv_id, paper_id, title, abstract, pdf_url, published_date, categories, pdf_processed\n"
        "- Node :Author properties: name\n"
        "- Node :Reference properties: title\n"
        "- Relationship (:Author)-[:WROTE]->(:Paper)\n"
        "- Relationship (:Paper)-[:CITES]->(:Reference)\n"
        "- Relationship (:Paper)-[:CITES_PAPER]->(:Paper)\n"
        "- Primary lookup key for papers is :Paper.arxiv_id\n"
        "- Paper titles are stored in :Paper.title\n"
        "- Reference nodes do not reliably have arxiv_id; title is the stable property\n"
    )


def _build_query_templates(intent: GraphRetrieveIntent) -> str:
    templates = [
        "Cypher templates you can adapt:",
        "1. Exact paper lookup by arXiv ID",
        "MATCH (p:Paper {arxiv_id: $arxiv_id})",
        "RETURN p.arxiv_id AS arxiv_id, p.title AS title, p.abstract AS abstract",
        "",
        "2. Lookup by arXiv ID with optional versionless fallback",
        "MATCH (p:Paper)",
        "WHERE p.arxiv_id = $arxiv_id OR p.arxiv_id = $arxiv_id_without_version",
        "RETURN p.arxiv_id AS arxiv_id, p.title AS title",
        "",
        "3. Paper lookup by title",
        "MATCH (p:Paper)",
        "WHERE toLower(p.title) CONTAINS toLower($title)",
        "   OR toLower($title) CONTAINS toLower(p.title)",
        "RETURN p.arxiv_id AS arxiv_id, p.title AS title, p.abstract AS abstract",
        "LIMIT 25",
        "",
        "4. Authors of a paper",
        "MATCH (a:Author)-[:WROTE]->(p:Paper)",
        "WHERE p.arxiv_id = $arxiv_id",
        "RETURN p.title AS title, collect(a.name) AS authors",
        "",
        "5. Shared citations between two papers",
        "MATCH (a:Paper), (b:Paper)",
        "WHERE a.arxiv_id = $arxiv_id_a AND b.arxiv_id = $arxiv_id_b",
        "MATCH (a)-[:CITES|CITES_PAPER]->(r)<-[:CITES|CITES_PAPER]-(b)",
        "RETURN DISTINCT r.title AS title, r.arxiv_id AS arxiv_id, labels(r) AS labels",
        "LIMIT 200",
        "",
        "Rules:",
        "- Use ONLY the labels, properties, and relationships listed in the schema summary.",
        "- Prefer exact arxiv_id match when the user provides an arXiv ID.",
        "- If the user asks for a Cypher/Neo4j query, answer with a runnable Cypher statement first.",
        "- Do not invent labels such as :Article or properties not shown above.",
    ]

    if intent.arxiv_ids:
        first_id = intent.arxiv_ids[0]
        versionless = first_id.split("v")[0] if "v" in first_id else first_id
        templates.extend(
            [
                "",
                "Suggested starting point for this question:",
                f"MATCH (p:Paper)",
                f"WHERE p.arxiv_id = '{first_id}' OR p.arxiv_id = '{versionless}'",
                "RETURN p.arxiv_id AS arxiv_id, p.title AS title",
                "LIMIT 1",
            ]
        )
    elif len(intent.paper_titles) >= 2 and intent.query_kind == "shared_citations":
        templates.extend(
            [
                "",
                "Suggested starting point for this question:",
                "MATCH (a:Paper), (b:Paper)",
                f"WHERE toLower(a.title) CONTAINS toLower('{intent.paper_titles[0]}')",
                f"  AND toLower(b.title) CONTAINS toLower('{intent.paper_titles[1]}')",
                "MATCH (a)-[:CITES|CITES_PAPER]->(r)<-[:CITES|CITES_PAPER]-(b)",
                "RETURN DISTINCT r.title AS title, r.arxiv_id AS arxiv_id",
                "LIMIT 200",
            ]
        )
    elif intent.paper_titles:
        templates.extend(
            [
                "",
                "Suggested starting point for this question:",
                "MATCH (p:Paper)",
                f"WHERE toLower(p.title) CONTAINS toLower('{intent.paper_titles[0]}')",
                "RETURN p.arxiv_id AS arxiv_id, p.title AS title, p.abstract AS abstract",
                "LIMIT 5",
            ]
        )

    return "\n".join(templates)


def _build_graph_guidance_doc(query: str, intent: GraphRetrieveIntent) -> Dict[str, Any]:
    context = "\n\n".join(
        [
            "This question should be handled as a Neo4j graph query generation task.",
            f"Detected graph query kind: {intent.query_kind}",
            f"Original user question: {query}",
            f"Extracted arXiv IDs: {intent.arxiv_ids or []}",
            f"Extracted quoted paper titles: {intent.paper_titles or []}",
            _build_schema_summary(),
            _build_query_templates(intent),
        ]
    )
    return {
        "page_content": context,
        "metadata": {
            "arxiv_id": intent.arxiv_ids[0] if intent.arxiv_ids else "",
            "title": "Neo4j graph query guidance",
            "authors": [],
            "score": 1.0,
            "source": _build_pdf_url(intent.arxiv_ids[0]) if intent.arxiv_ids else "",
            "section": "neo4j_graph_query_guidance",
            "search_mode": "neo4j_graph",
            "top_k": 1,
            "graph_query_kind": intent.query_kind,
            "graph_query_generation": True,
            "graph_query_schema_labels": ["Paper", "Author", "Reference"],
        },
    }


def _lookup_paper_by_arxiv_id(neo4j_client: Any, arxiv_id: str) -> Optional[Dict[str, Any]]:
    versionless = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
    rows = neo4j_client.execute_read(
        """
        MATCH (p:Paper)
        WHERE p.arxiv_id = $arxiv_id OR p.arxiv_id = $versionless
        RETURN p.arxiv_id AS arxiv_id,
               p.title AS title,
               p.abstract AS abstract,
               p.pdf_url AS pdf_url
        LIMIT 1
        """,
        {"arxiv_id": arxiv_id, "versionless": versionless},
    )
    return rows[0] if rows else None


def _lookup_paper_by_title(neo4j_client: Any, title: str) -> Optional[Dict[str, Any]]:
    normalized = re.sub(r"\s+", " ", title).strip().lower()
    rows = neo4j_client.execute_read(
        """
        MATCH (p:Paper)
        WHERE toLower(p.title) = $normalized_title
           OR toLower(p.title) CONTAINS $normalized_title
           OR $normalized_title CONTAINS toLower(p.title)
        RETURN p.arxiv_id AS arxiv_id,
               p.title AS title,
               p.abstract AS abstract,
               p.pdf_url AS pdf_url
        LIMIT 5
        """,
        {"normalized_title": normalized},
    )
    return rows[0] if rows else None


def _lookup_shared_citations_by_titles(
    neo4j_client: Any,
    title_a: str,
    title_b: str,
) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    paper_a = _lookup_paper_by_title(neo4j_client, title_a)
    paper_b = _lookup_paper_by_title(neo4j_client, title_b)
    if not paper_a or not paper_b:
        return paper_a, paper_b, []

    rows = neo4j_client.execute_read(
        """
        MATCH (a:Paper {arxiv_id: $arxiv_id_a}), (b:Paper {arxiv_id: $arxiv_id_b})
        MATCH (a)-[:CITES|CITES_PAPER]->(r)<-[:CITES|CITES_PAPER]-(b)
        RETURN DISTINCT r.title AS title,
               r.arxiv_id AS arxiv_id,
               labels(r) AS labels
        LIMIT 200
        """,
        {
            "arxiv_id_a": paper_a["arxiv_id"],
            "arxiv_id_b": paper_b["arxiv_id"],
        },
    )
    return paper_a, paper_b, rows


def _build_paper_lookup_doc(question: str, paper: Dict[str, Any]) -> Dict[str, Any]:
    arxiv_id = paper.get("arxiv_id", "") or ""
    title = paper.get("title", "") or "Untitled paper"
    abstract = paper.get("abstract", "") or ""
    pdf_url = paper.get("pdf_url") or _build_pdf_url(arxiv_id)
    content_lines = [
        f"Question: {question}",
        f"Matched paper arXiv ID: {arxiv_id}",
        f"Paper title: {title}",
    ]
    if abstract:
        content_lines.append(f"Abstract: {abstract}")

    return {
        "page_content": "\n".join(content_lines),
        "metadata": {
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": [],
            "score": 1.0,
            "source": pdf_url,
            "section": "neo4j_paper_lookup",
            "search_mode": "neo4j_graph",
            "top_k": 1,
            "graph_query_generation": False,
        },
    }


def _build_shared_citation_docs(
    question: str,
    paper_a: Dict[str, Any],
    paper_b: Dict[str, Any],
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not rows:
        return [
            {
                "page_content": (
                    f"Question: {question}\n"
                    f"No shared citations found between '{paper_a.get('title', 'Unknown paper')}' "
                    f"and '{paper_b.get('title', 'Unknown paper')}'."
                ),
                "metadata": {
                    "arxiv_id": "",
                    "title": "No shared citations found",
                    "authors": [],
                    "score": 0.0,
                    "source": "",
                    "section": "neo4j_shared_citations",
                    "search_mode": "neo4j_graph",
                    "top_k": 0,
                    "graph_query_generation": False,
                    "source_papers": [
                        paper_a.get("arxiv_id", ""),
                        paper_b.get("arxiv_id", ""),
                    ],
                },
            }
        ]

    docs: List[Dict[str, Any]] = []
    for row in rows:
        cited_title = row.get("title") or "Untitled reference"
        cited_arxiv_id = row.get("arxiv_id") or ""
        docs.append(
            {
                "page_content": (
                    f"Question: {question}\n"
                    f"Shared citation cited by both '{paper_a.get('title', 'Unknown paper')}' and "
                    f"'{paper_b.get('title', 'Unknown paper')}': {cited_title}"
                    + (f" (arXiv:{cited_arxiv_id})" if cited_arxiv_id else "")
                ),
                "metadata": {
                    "arxiv_id": cited_arxiv_id,
                    "title": cited_title,
                    "authors": [],
                    "score": 1.0,
                    "source": _build_pdf_url(cited_arxiv_id),
                    "section": "neo4j_shared_citations",
                    "search_mode": "neo4j_graph",
                    "top_k": len(rows),
                    "graph_query_generation": False,
                    "source_papers": [
                        paper_a.get("arxiv_id", ""),
                        paper_b.get("arxiv_id", ""),
                    ],
                    "labels": row.get("labels", []),
                },
            }
        )
    return docs


def _build_not_found_doc(question: str, arxiv_id: str) -> Dict[str, Any]:
    return {
        "page_content": (
            f"Question: {question}\n"
            f"No Paper node was found in Neo4j for arXiv ID {arxiv_id}."
        ),
        "metadata": {
            "arxiv_id": arxiv_id,
            "title": "Paper not found in Neo4j",
            "authors": [],
            "score": 0.0,
            "source": _build_pdf_url(arxiv_id),
            "section": "neo4j_paper_lookup",
            "search_mode": "neo4j_graph",
            "top_k": 0,
            "graph_query_generation": False,
        },
    }


def _build_title_not_found_doc(question: str, title: str) -> Dict[str, Any]:
    return {
        "page_content": (
            f"Question: {question}\n"
            f"No Paper node was found in Neo4j for title '{title}'."
        ),
        "metadata": {
            "arxiv_id": "",
            "title": title,
            "authors": [],
            "score": 0.0,
            "source": "",
            "section": "neo4j_title_lookup",
            "search_mode": "neo4j_graph",
            "top_k": 0,
            "graph_query_generation": False,
        },
    }


async def ainvoke_graph_retrieve_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, Any]:
    """Prepare Neo4j schema/query guidance for graph-native questions.

    This node does not attempt to answer the graph question itself. It classifies
    whether the user is asking for a Neo4j/Cypher-style graph lookup and, when
    applicable, returns schema-aware guidance that the answer-generation LLM can
    use to produce a Cypher query directly.
    """

    query = get_latest_query(state["messages"])
    neo4j_client = runtime.context.neo4j_client
    if not neo4j_client:
        logger.info("Graph retrieve skipped: Neo4j client unavailable")
        return {
            "metadata": {
                "graph_query_applicable": False,
                "graph_query_reason": "neo4j_unavailable",
            }
        }

    intent = _classify_graph_query(query)
    if not intent.should_use_graph:
        logger.info("Graph retrieve skipped: query not suitable for graph-first retrieval")
        return {
            "metadata": {
                "graph_query_applicable": False,
                "graph_query_reason": intent.reasoning,
            }
        }

    logger.info(
        "Graph retrieve applicable: kind=%s arxiv_ids=%s titles=%s",
        intent.query_kind,
        intent.arxiv_ids,
        intent.paper_titles,
    )

    if intent.query_kind == "paper_lookup" and intent.arxiv_ids:
        matched_paper = _lookup_paper_by_arxiv_id(neo4j_client, intent.arxiv_ids[0])
        if matched_paper:
            doc = _build_paper_lookup_doc(query, matched_paper)
            return {
                "retrieved_docs": [doc],
                "relevant_sources": [
                    {
                        "arxiv_id": matched_paper.get("arxiv_id", ""),
                        "title": matched_paper.get("title", ""),
                        "authors": [],
                        "url": matched_paper.get("pdf_url") or _build_pdf_url(matched_paper.get("arxiv_id")),
                        "relevance_score": 1.0,
                    }
                ],
                "metadata": {
                    "graph_query_applicable": True,
                    "graph_query_reason": intent.reasoning,
                    "graph_retrieval_attempted": True,
                    "graph_retrieval_used": True,
                    "graph_enriched_docs": 1,
                    "graph_enriched_arxiv_ids": [matched_paper.get("arxiv_id", "")],
                    "graph_query_titles": intent.paper_titles,
                    "graph_query_type": intent.query_kind,
                    "graph_query_rows": 1,
                    "graph_query_satisfied": True,
                    "graph_query_generation": False,
                },
            }

        doc = _build_not_found_doc(query, intent.arxiv_ids[0])
        return {
            "retrieved_docs": [doc],
            "relevant_sources": [],
            "metadata": {
                "graph_query_applicable": True,
                "graph_query_reason": "paper_not_found",
                "graph_retrieval_attempted": True,
                "graph_retrieval_used": True,
                "graph_enriched_docs": 1,
                "graph_enriched_arxiv_ids": intent.arxiv_ids,
                "graph_query_titles": intent.paper_titles,
                "graph_query_type": intent.query_kind,
                "graph_query_rows": 0,
                "graph_query_satisfied": False,
                "graph_query_generation": False,
            },
        }

    if intent.query_kind == "title_to_arxiv_lookup" and intent.paper_titles:
        matched_paper = _lookup_paper_by_title(neo4j_client, intent.paper_titles[0])
        if matched_paper:
            doc = _build_paper_lookup_doc(query, matched_paper)
            return {
                "retrieved_docs": [doc],
                "relevant_sources": [
                    {
                        "arxiv_id": matched_paper.get("arxiv_id", ""),
                        "title": matched_paper.get("title", ""),
                        "authors": [],
                        "url": matched_paper.get("pdf_url") or _build_pdf_url(matched_paper.get("arxiv_id")),
                        "relevance_score": 1.0,
                    }
                ],
                "metadata": {
                    "graph_query_applicable": True,
                    "graph_query_reason": intent.reasoning,
                    "graph_retrieval_attempted": True,
                    "graph_retrieval_used": True,
                    "graph_enriched_docs": 1,
                    "graph_enriched_arxiv_ids": [matched_paper.get("arxiv_id", "")],
                    "graph_query_titles": intent.paper_titles,
                    "graph_query_type": intent.query_kind,
                    "graph_query_rows": 1,
                    "graph_query_satisfied": True,
                    "graph_query_generation": False,
                },
            }

        doc = _build_title_not_found_doc(query, intent.paper_titles[0])
        return {
            "retrieved_docs": [doc],
            "relevant_sources": [],
            "metadata": {
                "graph_query_applicable": True,
                "graph_query_reason": "paper_title_not_found",
                "graph_retrieval_attempted": True,
                "graph_retrieval_used": True,
                "graph_enriched_docs": 1,
                "graph_enriched_arxiv_ids": [],
                "graph_query_titles": intent.paper_titles,
                "graph_query_type": intent.query_kind,
                "graph_query_rows": 0,
                "graph_query_satisfied": False,
                "graph_query_generation": False,
            },
        }

    if intent.query_kind == "shared_citations" and len(intent.paper_titles) >= 2:
        paper_a, paper_b, rows = _lookup_shared_citations_by_titles(
            neo4j_client,
            intent.paper_titles[0],
            intent.paper_titles[1],
        )
        if paper_a and paper_b:
            docs = _build_shared_citation_docs(query, paper_a, paper_b, rows)
            return {
                "retrieved_docs": docs,
                "relevant_sources": [
                    {
                        "arxiv_id": paper_a.get("arxiv_id", ""),
                        "title": paper_a.get("title", ""),
                        "authors": [],
                        "url": paper_a.get("pdf_url") or _build_pdf_url(paper_a.get("arxiv_id")),
                        "relevance_score": 1.0,
                    },
                    {
                        "arxiv_id": paper_b.get("arxiv_id", ""),
                        "title": paper_b.get("title", ""),
                        "authors": [],
                        "url": paper_b.get("pdf_url") or _build_pdf_url(paper_b.get("arxiv_id")),
                        "relevance_score": 1.0,
                    },
                ],
                "metadata": {
                    "graph_query_applicable": True,
                    "graph_query_reason": intent.reasoning,
                    "graph_retrieval_attempted": True,
                    "graph_retrieval_used": True,
                    "graph_enriched_docs": len(docs),
                    "graph_enriched_arxiv_ids": [
                        paper_a.get("arxiv_id", ""),
                        paper_b.get("arxiv_id", ""),
                    ],
                    "graph_query_titles": intent.paper_titles,
                    "graph_query_type": intent.query_kind,
                    "graph_query_rows": len(rows),
                    "graph_query_satisfied": True,
                    "graph_query_generation": False,
                },
            }

        missing_title = intent.paper_titles[0] if not paper_a else intent.paper_titles[1]
        doc = _build_title_not_found_doc(query, missing_title)
        return {
            "retrieved_docs": [doc],
            "relevant_sources": [],
            "metadata": {
                "graph_query_applicable": True,
                "graph_query_reason": "paper_title_not_found",
                "graph_retrieval_attempted": True,
                "graph_retrieval_used": True,
                "graph_enriched_docs": 1,
                "graph_enriched_arxiv_ids": [],
                "graph_query_titles": intent.paper_titles,
                "graph_query_type": intent.query_kind,
                "graph_query_rows": 0,
                "graph_query_satisfied": False,
                "graph_query_generation": False,
            },
        }

    guidance_doc = _build_graph_guidance_doc(query, intent)
    relevant_sources = [
        {
            "arxiv_id": arxiv_id,
            "title": "Graph lookup target",
            "authors": [],
            "url": _build_pdf_url(arxiv_id),
            "relevance_score": 1.0,
        }
        for arxiv_id in intent.arxiv_ids
    ]

    return {
        "retrieved_docs": [guidance_doc],
        "relevant_sources": relevant_sources,
        "metadata": {
            "graph_query_applicable": True,
            "graph_query_reason": intent.reasoning,
            "graph_retrieval_attempted": True,
            "graph_retrieval_used": True,
            "graph_enriched_docs": 1,
            "graph_enriched_arxiv_ids": intent.arxiv_ids,
            "graph_query_titles": intent.paper_titles,
            "graph_query_type": intent.query_kind,
            "graph_query_rows": 0,
            "graph_query_satisfied": True,
            "graph_query_generation": True,
        },
    }


def route_after_graph_retrieve(state: AgentState) -> str:
    """Route graph-first retrieval to rerank or fallback retrieval."""

    metadata = state.get("metadata", {}) or {}
    if metadata.get("graph_query_applicable"):
        if metadata.get("graph_retrieval_used") and not metadata.get("graph_query_generation", False):
            logger.info("Graph retrieve returned factual Neo4j results -> skip rerank, go to grade_documents")
            return "grade_documents"
        logger.info("Graph retrieve applicable -> rerank graph guidance")
        return "rerank"

    logger.info("Graph retrieve not applicable -> fallback to standard retrieve")
    return "retrieve"

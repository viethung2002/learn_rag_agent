import asyncio
from types import SimpleNamespace

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from src.services.agents.agentic_rag import extract_quoted_titles, is_shared_citation_query
from src.services.agents.nodes.graph_retrieve_node import (
    ainvoke_graph_retrieve_step,
    _build_paper_lookup_doc,
    _classify_graph_query,
    route_after_graph_retrieve,
)
from src.services.agents.nodes.rerank_documents_node import ainvoke_rerank_documents_step
from src.services.agents.nodes.generate_answer_node import _build_context_from_retrieved_docs as build_answer_context
from src.services.agents.nodes.grade_documents_node import _build_context_from_retrieved_docs as build_grade_context


def test_extract_quoted_titles_handles_double_quotes():
    query = (
        'Which bibliography entries are cited by both the paper '
        '"Light-ResKAN: A Parameter-Sharing Lightweight KAN with Gram Polynomials for Efficient SAR Image Recognition" '
        'and the paper "Deep Residual Learning for Image Recognition"?'
    )

    assert extract_quoted_titles(query) == [
        "Light-ResKAN: A Parameter-Sharing Lightweight KAN with Gram Polynomials for Efficient SAR Image Recognition",
        "Deep Residual Learning for Image Recognition",
    ]


def test_is_shared_citation_query_detects_both_papers_pattern():
    query = (
        'Which bibliography entries are cited by both the paper '
        '"Paper A" and the paper "Paper B"?'
    )

    assert is_shared_citation_query(query) is True


def test_is_shared_citation_query_rejects_single_paper_query():
    query = 'Summarize the paper "Deep Residual Learning for Image Recognition".'

    assert is_shared_citation_query(query) is False


def test_classify_graph_query_detects_arxiv_id_lookup():
    query = "What is the title of the paper with arXiv ID 1512.03385v1?"

    result = _classify_graph_query(query)

    assert result.should_use_graph is True
    assert result.query_kind == "paper_lookup"
    assert result.arxiv_ids == ["1512.03385v1"]


def test_classify_graph_query_detects_explicit_cypher_request():
    query = "Cho toi cau lenh Neo4j de truy van title theo arXiv ID 1512.03385v1"

    result = _classify_graph_query(query)

    assert result.should_use_graph is True
    assert result.query_kind == "cypher_generation"


def test_build_paper_lookup_doc_uses_real_paper_title():
    doc = _build_paper_lookup_doc(
        "What is the title of the paper with arXiv ID 1512.03385v1?",
        {
            "arxiv_id": "1512.03385v1",
            "title": "Deep Residual Learning for Image Recognition",
            "abstract": "Residual networks improve training.",
            "pdf_url": "https://arxiv.org/pdf/1512.03385.pdf",
        },
    )

    assert doc["metadata"]["title"] == "Deep Residual Learning for Image Recognition"
    assert "Neo4j graph query guidance" not in doc["page_content"]


def test_classify_graph_query_detects_title_to_arxiv_lookup():
    query = "What is the arXiv ID of the article 'AN ANALYSIS OF DEEP NEURAL NETWORK MODELS FOR PRACTICAL APPLICATIONS'?"

    result = _classify_graph_query(query)

    assert result.should_use_graph is True
    assert result.query_kind == "title_to_arxiv_lookup"
    assert result.paper_titles == ["AN ANALYSIS OF DEEP NEURAL NETWORK MODELS FOR PRACTICAL APPLICATIONS"]


def test_graph_retrieve_executes_shared_citation_lookup_instead_of_returning_cypher_guidance():
    class FakeNeo4jClient:
        def execute_read(self, query, params):
            normalized = " ".join(query.split())
            if "RETURN p.arxiv_id AS arxiv_id" in normalized:
                title = params["normalized_title"]
                if "deep residual learning for image recognition" in title:
                    return [
                        {
                            "arxiv_id": "1512.03385",
                            "title": "Deep Residual Learning for Image Recognition",
                            "abstract": "Residual learning.",
                            "pdf_url": "https://arxiv.org/pdf/1512.03385.pdf",
                        }
                    ]
                if "an analysis of deep neural network models for practical applications" in title:
                    return [
                        {
                            "arxiv_id": "2401.00001",
                            "title": "An Analysis of Deep Neural Network Models for Practical Applications",
                            "abstract": "Model analysis.",
                            "pdf_url": "https://arxiv.org/pdf/2401.00001.pdf",
                        }
                    ]
                return []

            if "MATCH (a:Paper {arxiv_id: $arxiv_id_a}), (b:Paper {arxiv_id: $arxiv_id_b})" in normalized:
                return [
                    {
                        "title": "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift",
                        "arxiv_id": "1502.03167",
                        "labels": ["Paper"],
                    }
                ]

            raise AssertionError(f"Unexpected query: {query}")

    runtime = SimpleNamespace(
        context=SimpleNamespace(
            neo4j_client=FakeNeo4jClient(),
        )
    )
    state = {
        "messages": [
            HumanMessage(
                content=(
                    "Do the papers 'Deep Residual Learning for Image Recognition' and "
                    "'An Analysis of Deep Neural Network Models for Practical Applications' "
                    "share any common references?"
                )
            )
        ]
    }

    result = asyncio.run(ainvoke_graph_retrieve_step(state, runtime))

    assert result["metadata"]["graph_query_type"] == "shared_citations"
    assert result["metadata"]["graph_query_generation"] is False
    assert "Neo4j graph query guidance" not in result["retrieved_docs"][0]["page_content"]
    assert "Batch Normalization" in result["retrieved_docs"][0]["page_content"]


def test_route_after_graph_retrieve_skips_rerank_for_graph_only_results():
    state = {
        "metadata": {
            "graph_query_applicable": True,
            "graph_retrieval_used": True,
            "graph_query_generation": False,
        }
    }

    assert route_after_graph_retrieve(state) == "grade_documents"


def test_rerank_node_skips_rerank_for_neo4j_graph_documents():
    class FakeNvidiaClient:
        def get_reranker(self, **kwargs):
            raise AssertionError("Reranker should not be called for graph-only results")

    docs = [
        {
            "page_content": "Shared citation A",
            "metadata": {"search_mode": "neo4j_graph", "relevance_score": 1.0, "title": "A"},
        },
        {
            "page_content": "Shared citation B",
            "metadata": {"search_mode": "neo4j_graph", "relevance_score": 1.0, "title": "B"},
        },
        {
            "page_content": "Shared citation C",
            "metadata": {"search_mode": "neo4j_graph", "relevance_score": 1.0, "title": "C"},
        },
        {
            "page_content": "Shared citation D",
            "metadata": {"search_mode": "neo4j_graph", "relevance_score": 1.0, "title": "D"},
        },
    ]
    state = {
        "messages": [HumanMessage(content="shared references?")],
        "retrieved_docs": docs,
    }
    runtime = SimpleNamespace(
        context=SimpleNamespace(
            nvidia_client=FakeNvidiaClient(),
        )
    )

    result = asyncio.run(ainvoke_rerank_documents_step(state, runtime))

    assert len(result["retrieved_docs"]) == 4
    assert all(isinstance(doc, Document) for doc in result["retrieved_docs"])


def test_context_builders_use_retrieved_docs_when_no_tool_message_exists():
    state = {
        "messages": [HumanMessage(content="shared references?")],
        "retrieved_docs": [
            {
                "page_content": "Shared citation cited by both papers: Going deeper with convolutions",
                "metadata": {"search_mode": "neo4j_graph"},
            },
            {
                "page_content": "Shared citation cited by both papers: Network in network",
                "metadata": {"search_mode": "neo4j_graph"},
            },
        ],
    }

    answer_context = build_answer_context(state)
    grade_context = build_grade_context(state)

    assert "Going deeper with convolutions" in answer_context
    assert "Network in network" in answer_context
    assert answer_context == grade_context

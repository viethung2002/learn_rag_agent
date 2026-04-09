from src.services.agents.agentic_rag import extract_quoted_titles, is_shared_citation_query
from src.services.agents.nodes.graph_retrieve_node import (
    _build_paper_lookup_doc,
    _classify_graph_query,
)


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

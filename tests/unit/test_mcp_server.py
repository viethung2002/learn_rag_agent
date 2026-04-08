from langchain_core.documents import Document

from src.mcp.mcp_server import _heuristic_route, merge_routed_results


def test_heuristic_route_prefers_graph_for_shared_citations():
    decision = _heuristic_route(
        'Which shared citations do "Attention Is All You Need" and "BERT: Pre-training of Deep Bidirectional Transformers" have?'
    )

    assert decision.action == "graph"
    assert decision.graph_enabled is True
    assert decision.search_enabled is False


def test_heuristic_route_uses_both_when_graph_and_similarity_are_requested():
    decision = _heuristic_route(
        'Find papers similar to graph neural networks for drug discovery and also show the citation network around the main papers.'
    )

    assert decision.action == "both"
    assert decision.graph_enabled is True
    assert decision.search_enabled is True


def test_merge_routed_results_deduplicates_by_arxiv_id_and_keeps_provenance():
    search_doc = Document(
        page_content="Semantic summary",
        metadata={
            "arxiv_id": "1234.5678",
            "title": "Graph Learning",
            "score": 0.82,
            "provenance": "opensearch",
            "section": "abstract",
        },
    )
    graph_doc = Document(
        page_content="Graph relation summary with more detail",
        metadata={
            "arxiv_id": "1234.5678",
            "title": "Graph Learning",
            "score": 1.0,
            "provenance": "neo4j",
            "section": "shared_citation",
        },
    )

    merged = merge_routed_results(
        {
            "retrieve_papers": [search_doc],
            "graph_citation_retrieve": [graph_doc],
        }
    )

    assert len(merged) == 1
    item = merged[0]
    assert item["arxiv_id"] == "1234.5678"
    assert item["score"] == 1.0
    assert item["page_content"] == "Graph relation summary with more detail"
    assert set(item["sources"]) == {"opensearch", "neo4j"}
    assert set(item["tool_names"]) == {"retrieve_papers", "graph_citation_retrieve"}

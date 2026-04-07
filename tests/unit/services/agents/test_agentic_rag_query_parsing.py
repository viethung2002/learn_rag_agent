from src.services.agents.agentic_rag import extract_quoted_titles, is_shared_citation_query


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

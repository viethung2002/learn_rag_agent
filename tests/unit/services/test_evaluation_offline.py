from src.services.evaluation.offline import SyntheticDatasetGenerator


def test_map_ragas_question_type() -> None:
    assert SyntheticDatasetGenerator._map_ragas_question_type("single_hop_specific_query_synthesizer") == "simple"
    assert SyntheticDatasetGenerator._map_ragas_question_type("multi_hop_specific_query_synthesizer") == "reasoning"
    assert SyntheticDatasetGenerator._map_ragas_question_type("multi_hop_abstract_query_synthesizer") == "multi_context"


def test_convert_ragas_row_to_project_schema() -> None:
    generator = SyntheticDatasetGenerator(
        opensearch_client=None,
        session=None,
        generator_provider="ollama",
        generator_model="llama3.2:3b",
    )
    context_lookup = {
        "ctx-a": {
            "source_kind": "neo4j",
            "arxiv_ids": ["1234.5678", "9999.0001"],
            "sample_metadata": {"relation_kind": "citation_pair"},
        },
        "ctx-b": {
            "source_kind": "neo4j",
            "arxiv_ids": ["1234.5678", "9999.0001"],
            "sample_metadata": {"author_name": "Ada"},
        },
    }
    row = {
        "user_input": "How are the two papers related?",
        "reference": "They are connected through citation evidence.",
        "reference_contexts": ["ctx-a", "ctx-b"],
        "synthesizer_name": "multi_hop_abstract_query_synthesizer",
        "persona_name": "Researcher",
        "query_style": "Formal",
        "query_length": "Medium",
    }

    result = generator._convert_ragas_row(row=row, context_lookup=context_lookup)

    assert result is not None
    assert result["question"] == "How are the two papers related?"
    assert result["ground_truth"] == "They are connected through citation evidence."
    assert result["question_type"] == "multi_context"
    assert result["difficulty"] == "hard"
    assert result["source_kind"] == "neo4j"
    assert result["arxiv_ids"] == ["1234.5678", "9999.0001"]
    assert result["metadata"]["synthesizer_name"] == "multi_hop_abstract_query_synthesizer"
    assert result["metadata"]["relation_kind"] == "citation_pair"
    assert result["metadata"]["author_name"] == "Ada"

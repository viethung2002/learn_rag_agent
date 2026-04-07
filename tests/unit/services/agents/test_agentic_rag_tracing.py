import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from langchain_core.messages import AIMessage, HumanMessage

from src.services.agents.agentic_rag import AgenticRAGService
from src.services.agents.models import GuardrailScoring


def test_run_workflow_does_not_end_context_managed_trace():
    service = AgenticRAGService.__new__(AgenticRAGService)
    service.graph = Mock()
    service.graph.ainvoke = AsyncMock(
        return_value={
            "messages": [HumanMessage(content="test query"), AIMessage(content="test answer")],
            "retrieval_attempts": 1,
            "guardrail_result": GuardrailScoring(score=90, reason="In scope"),
            "relevant_sources": [{"url": "https://arxiv.org/abs/1234.5678"}],
            "grading_results": [],
            "retrieved_docs": [],
            "metadata": {},
            "rewritten_query": None,
        }
    )
    service.graph_config = SimpleNamespace(
        temperature=0.0,
        top_k=3,
        max_retrieval_attempts=2,
        guardrail_threshold=60,
    )
    service.nvidia = Mock()
    service.opensearch = Mock()
    service.neo4j = None
    service.embeddings = Mock()
    service.tools_by_name = {}

    tracer = Mock()
    tracer.client = Mock()
    tracer.get_trace_id.return_value = "trace-123"
    service.langfuse_tracer = tracer

    trace = Mock()

    result = asyncio.run(
        service._run_workflow(
            query="test query",
            model_to_use="test-model",
            user_id="user-1",
            trace=trace,
            thread_id="thread-1",
        )
    )

    assert result["trace_id"] == "trace-123"
    trace.update.assert_called_once()
    trace.end.assert_not_called()
    tracer.flush.assert_called_once()

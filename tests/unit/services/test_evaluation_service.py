import asyncio
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

from src.config import EvaluationSettings, Settings
from src.schemas.api.ask import LLMJudgeResult
from src.services.evaluation.service import EvaluationService


def make_service(**evaluation_overrides) -> EvaluationService:
    settings = Settings(
        _env_file=None,
        debug=True,
        evaluation=EvaluationSettings(_env_file=None, **evaluation_overrides),
    )
    return EvaluationService(settings=settings)


def test_evaluate_answer_skips_when_reference_required() -> None:
    service = make_service(enabled=True, require_reference=True)
    service._run_ragas = AsyncMock()
    service._run_llm_judge = AsyncMock()

    result = asyncio.run(
        service.evaluate_answer(
            query="What is RAG?",
            answer="Retrieval-augmented generation.",
            contexts=["ctx1", "ctx2"],
        )
    )

    assert result is not None
    assert result.status == "skipped_missing_reference"
    assert result.reference_used is False
    assert result.ragas_metrics == {}
    assert result.llm_judge is None
    service._run_ragas.assert_not_called()
    service._run_llm_judge.assert_not_called()


def test_evaluate_answer_runs_ragas_and_judge_with_trimmed_contexts() -> None:
    service = make_service(enabled=True, max_contexts=2, require_reference=False)
    service._run_ragas = AsyncMock(return_value={"faithfulness": 0.91})
    service._run_llm_judge = AsyncMock(
        return_value=LLMJudgeResult(score=0.8, verdict="good", reasoning="Grounded in context")
    )

    result = asyncio.run(
        service.evaluate_answer(
            query="What is RAG?",
            answer="Retrieval-augmented generation.",
            contexts=["ctx1", "ctx2", "ctx3"],
            reference_answer="RAG combines retrieval with generation.",
        )
    )

    assert result is not None
    assert result.status == "completed"
    assert result.reference_used is True
    assert result.ragas_metrics == {"faithfulness": 0.91}
    assert result.llm_judge is not None
    assert result.llm_judge.score == 0.8

    service._run_ragas.assert_awaited_once_with(
        query="What is RAG?",
        answer="Retrieval-augmented generation.",
        contexts=["ctx1", "ctx2"],
        reference_answer="RAG combines retrieval with generation.",
    )
    service._run_llm_judge.assert_awaited_once_with(
        query="What is RAG?",
        answer="Retrieval-augmented generation.",
        contexts=["ctx1", "ctx2"],
        reference_answer="RAG combines retrieval with generation.",
        judge_provider_override=None,
        judge_model_override=None,
    )


def test_run_ragas_uses_explicit_nvidia_langchain_model(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeDataset:
        @staticmethod
        def from_dict(rows):
            captured["rows"] = rows
            return rows

    class FakeResult:
        def to_pandas(self):
            return SimpleNamespace(
                iloc=[SimpleNamespace(to_dict=lambda: {"faithfulness": 0.75, "answer_relevancy": None})]
            )

    def fake_evaluate(*, dataset, metrics, llm, raise_exceptions):
        captured["dataset"] = dataset
        captured["metrics"] = metrics
        captured["llm"] = llm
        captured["raise_exceptions"] = raise_exceptions
        return FakeResult()

    datasets_module = ModuleType("datasets")
    datasets_module.Dataset = FakeDataset

    ragas_module = ModuleType("ragas")
    ragas_module.evaluate = fake_evaluate

    ragas_metrics_module = ModuleType("ragas.metrics")
    ragas_metrics_module.answer_relevancy = "answer_relevancy"
    ragas_metrics_module.context_precision = "context_precision"
    ragas_metrics_module.faithfulness = "faithfulness"

    ragas_llms_module = ModuleType("ragas.llms")

    class FakeLangchainLLMWrapper:
        def __init__(self, model):
            self.model = model

    ragas_llms_module.LangchainLLMWrapper = FakeLangchainLLMWrapper

    monkeypatch.setitem(sys.modules, "datasets", datasets_module)
    monkeypatch.setitem(sys.modules, "ragas", ragas_module)
    monkeypatch.setitem(sys.modules, "ragas.metrics", ragas_metrics_module)
    monkeypatch.setitem(sys.modules, "ragas.llms", ragas_llms_module)

    service = make_service(enabled=True)
    fake_model = object()
    service.nvidia_client = SimpleNamespace(
        get_langchain_model=lambda model, temperature: fake_model,
    )

    result = asyncio.run(
        service._run_ragas(
            query="What is RAG?",
            answer="Retrieval-augmented generation.",
            contexts=["ctx1"],
        )
    )

    assert result == {"faithfulness": 0.75}
    assert captured["rows"] == {
        "question": ["What is RAG?"],
        "answer": ["Retrieval-augmented generation."],
        "contexts": [["ctx1"]],
    }
    assert captured["metrics"] == ["faithfulness", "answer_relevancy", "context_precision"]
    assert isinstance(captured["llm"], FakeLangchainLLMWrapper)
    assert captured["llm"].model is fake_model
    assert captured["raise_exceptions"] is False


def test_run_ragas_skips_when_nvidia_client_is_missing() -> None:
    service = make_service(enabled=True)

    result = asyncio.run(
        service._run_ragas(
            query="What is RAG?",
            answer="Retrieval-augmented generation.",
            contexts=["ctx1"],
        )
    )

    assert result == {}

import json
import logging
from typing import Any, Dict, List, Optional

from src.config import Settings
from src.schemas.api.ask import EvaluationSummary, LLMJudgeResult
from src.services.langfuse.client import LangfuseTracer
from src.services.nvidia.client import NvidiaClient


logger = logging.getLogger(__name__)


class EvaluationService:
    """Optional RAG evaluation service for RAGAS and LLM-as-judge."""

    def __init__(
        self,
        settings: Settings,
        nvidia_client: NvidiaClient | None = None,
        langfuse_tracer: LangfuseTracer | None = None,
    ) -> None:
        self.settings = settings
        self.config = settings.evaluation
        self.nvidia_client = nvidia_client
        self.langfuse_tracer = langfuse_tracer

    async def evaluate_answer(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        reference_answer: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        judge_provider_override: Optional[str] = None,
        judge_model_override: Optional[str] = None,
    ) -> EvaluationSummary | None:
        """Run automatic evaluation for one RAG answer."""
        if not self.config.enabled:
            return None

        trimmed_contexts = [context for context in contexts[: self.config.max_contexts] if context]
        reference_used = bool(reference_answer)

        if self.config.require_reference and not reference_answer:
            return EvaluationSummary(
                ragas_metrics={},
                llm_judge=None,
                reference_used=False,
                status="skipped_missing_reference",
            )

        ragas_metrics: Dict[str, float] = {}
        llm_judge: LLMJudgeResult | None = None

        if self.config.run_ragas:
            ragas_metrics = await self._run_ragas(
                query=query,
                answer=answer,
                contexts=trimmed_contexts,
                reference_answer=reference_answer,
            )

        if self.config.run_llm_judge:
            llm_judge = await self._run_llm_judge(
                query=query,
                answer=answer,
                contexts=trimmed_contexts,
                reference_answer=reference_answer,
                judge_provider_override=judge_provider_override,
                judge_model_override=judge_model_override,
            )

        summary = EvaluationSummary(
            ragas_metrics=ragas_metrics,
            llm_judge=llm_judge,
            reference_used=reference_used,
            status="completed",
        )

        self._log_to_langfuse(summary=summary, query=query, metadata=metadata)
        return summary

    async def generate_testset_from_texts(
        self,
        texts: List[str],
        test_size: int = 50,
        distributions: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Generate a synthetic testset from a list of texts using RAGAS TestsetGenerator when available.

        Returns a list of dicts with keys: question, ground_truth, and optional metadata.
        """
        if not texts:
            return []

        # Prefer RAGAS TestsetGenerator for diverse item types
        if RAGAS_TESTSET_AVAILABLE:
            try:
                # Create generator; let user environment determine provider (OpenAI keys, etc.)
                gen = TestsetGenerator()

                # RAGAS offers generation helpers; prefer generate_with_langchain_docs when available
                if hasattr(gen, "generate_with_langchain_docs"):
                    # langchain docs expect list of Document objects or plain texts
                    testset = gen.generate_with_langchain_docs(
                        texts,
                        test_size=test_size,
                        distributions=distributions or {},
                    )
                    # Convert to list of dicts
                    try:
                        df = testset.to_pandas()
                        records = []
                        for row in df.to_dict(orient="records"):
                            records.append({
                                "question": row.get("question") or row.get("prompt") or "",
                                "ground_truth": row.get("answer") or row.get("ground_truth") or "",
                                "meta": row,
                            })
                        return records
                    except Exception:
                        # Fallback if not a dataset-like object
                        return [{"question": r.get("question"), "ground_truth": r.get("answer"), "meta": {}} for r in testset]

                # Fallback: try simple generate API
                items = gen.generate(texts, test_size=test_size)
                return items

            except Exception as exc:
                logger.warning("RAGAS TestsetGenerator failed: %s; falling back to LLM prompts", exc)

        # Fallback: use existing LLM clients to prompt-generate QA pairs
        results: List[Dict[str, Any]] = []
        for text in texts[:test_size]:
            prompt = (
                "You are an expert reviewer. Given the following passage from a scientific paper, create a single natural question whose answer is fully contained in the passage, "
                "and provide a concise, factual ground truth answer. Return JSON: {\"question\": ..., \"ground_truth\": ...}.\n\nPassage:\n"
                + text
            )
            try:
                response = None
                if self.nvidia_client:
                    resp = await self.nvidia_client.generate(model=self.settings.nvidia_model, prompt=prompt, temperature=0.0)
                    # Nvidia client returns {'parsed': ..., 'text': ...} or {'text': '...'}
                    if resp is None:
                        response = None
                    elif isinstance(resp, dict) and resp.get("parsed") is not None:
                        # prefer structured parsed result
                        response = json.dumps(resp.get("parsed"))
                    else:
                        response = resp.get("text") if isinstance(resp, dict) else None

                if not response:
                    continue

                parsed = None
                try:
                    parsed = json.loads(response)
                except Exception:
                    # try to extract JSON substring
                    import re

                    m = re.search(r"\{.*\}", response, re.DOTALL)
                    if m:
                        try:
                            parsed = json.loads(m.group(0))
                        except Exception:
                            parsed = None

                if parsed and parsed.get("question") and parsed.get("ground_truth"):
                    results.append({"question": parsed["question"], "ground_truth": parsed["ground_truth"], "meta": {}})
            except Exception as exc:
                logger.warning("LLM-based testset generation failed for one text: %s", exc)

        return results
    async def _run_ragas(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        reference_answer: Optional[str] = None,
    ) -> Dict[str, float]:
        """Run RAGAS metrics with an explicitly configured LangChain LLM."""
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import answer_relevancy, context_precision, faithfulness
            from ragas.llms import LangchainLLMWrapper

            rows = {
                "question": [query],
                "answer": [answer],
                "contexts": [contexts],
            }

            metrics = [faithfulness, answer_relevancy, context_precision]

            if reference_answer:
                from ragas.metrics import answer_correctness

                rows["ground_truth"] = [reference_answer]
                metrics.append(answer_correctness)

            dataset = Dataset.from_dict(rows)

            if not self.nvidia_client:
                logger.warning("NvidiaClient not initialized; skipping RAGAS evaluation to avoid provider fallback")
                return {}

            lc_model = None
            if hasattr(self.nvidia_client, "get_langchain_model"):
                try:
                    lc_model = self.nvidia_client.get_langchain_model(
                        model=self.settings.nvidia_model,
                        temperature=0.0,
                    )
                except Exception as exc:
                    logger.warning("Failed to build LangChain model for RAGAS: %s", exc)
                    return {}

            if lc_model is None:
                lc_model = getattr(self.nvidia_client, "model", None) or getattr(self.nvidia_client, "client", None)

            if lc_model is None:
                logger.warning("NvidiaClient has no LangChain model instance; skipping RAGAS evaluation")
                return {}

            ragas_llm = LangchainLLMWrapper(lc_model)
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=ragas_llm,
                raise_exceptions=False,
            )

            scores = result.to_pandas().iloc[0].to_dict()
            return {key: float(value) for key, value in scores.items() if value is not None}

        except ImportError:
            logger.warning("RAGAS dependencies are not installed; skipping RAGAS evaluation")
            return {}
        except Exception as exc:
            logger.warning("RAGAS evaluation failed: %s", exc)
            return {}

    async def _run_llm_judge(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        reference_answer: Optional[str] = None,
        judge_provider_override: Optional[str] = None,
        judge_model_override: Optional[str] = None,
    ) -> LLMJudgeResult | None:
        prompt = self._build_judge_prompt(
            query=query,
            answer=answer,
            contexts=contexts,
            reference_answer=reference_answer,
        )

        raw_text = None
        try:
            model = judge_model_override or self.config.judge_model

            if not self.nvidia_client:
                return None

            response = await self.nvidia_client.generate(
                model=model,
                prompt=prompt,
                temperature=0.0,
                top_p=0.1,
            )

            if not response:
                return None

            if isinstance(response, dict) and response.get("parsed") is not None:
                parsed = response.get("parsed")
            else:
                raw_text = response.get("text") if isinstance(response, dict) else None
                if not raw_text:
                    return None
                try:
                    parsed = json.loads(raw_text)
                except Exception:
                    # try to extract JSON substring
                    import re

                    m = re.search(r"\{.*\}", raw_text, re.DOTALL)
                    if not m:
                        return None
                    parsed = json.loads(m.group(0))
            score = max(0.0, min(1.0, float(parsed["score"])))
            return LLMJudgeResult(
                score=score,
                verdict=str(parsed["verdict"]),
                reasoning=str(parsed.get("reasoning") or ""),
            )
        except Exception as exc:
            logger.warning("LLM judge evaluation failed: %s", exc)
            return None

    def _build_judge_prompt(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        reference_answer: Optional[str] = None,
    ) -> str:
        context_block = "\n\n".join(f"[Context {idx + 1}]\n{context}" for idx, context in enumerate(contexts)) or "[No context]"
        reference_block = reference_answer or "None"
        return (
            "You are evaluating a RAG answer.\n"
            "Return JSON with keys: score, verdict, reasoning.\n"
            "Score must be between 0 and 1.\n"
            "Judge factual consistency with the provided context, relevance to the question, and completeness.\n\n"
            f"Question:\n{query}\n\n"
            f"Answer:\n{answer}\n\n"
            f"Reference Answer:\n{reference_block}\n\n"
            f"Retrieved Contexts:\n{context_block}\n"
        )

    def _log_to_langfuse(
        self,
        summary: EvaluationSummary,
        query: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.langfuse_tracer or not self.langfuse_tracer.client:
            return

        try:
            with self.langfuse_tracer.start_span(
                name="rag_evaluation",
                input_data={"query": query},
                metadata=metadata or {},
            ) as span:
                self.langfuse_tracer.update_span(
                    span=span,
                    output=summary.model_dump(),
                )
        except Exception as exc:
            logger.warning("Failed to log evaluation to Langfuse: %s", exc)

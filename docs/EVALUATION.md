# Evaluation Integration Guide

This project already has the main building blocks for `RAGAS` and `LLM-as-judge`.
The correct integration path is the new evaluation stack under:

- `src/services/evaluation/service.py`
- `src/services/evaluation/factory.py`
- `src/services/evaluation/offline.py`
- `scripts/generate_eval_dataset.py`
- `scripts/run_eval.py`

Do not extend the older compatibility path in:

- `src/services/evaluator.py`
- `src/routers/evaluate.py`

Those files are legacy and are not the main path used by the current `/api/v1/ask` and `/api/v1/ask-agentic` flows.

## 1. What is already integrated

The current code already wires evaluation into the live RAG endpoints:

- `src/main.py`
  - creates `app.state.evaluation_service` with `make_evaluation_service(...)`
- `src/dependencies.py`
  - exposes `EvaluationDep`
- `src/routers/ask.py`
  - calls `evaluation_service.evaluate_answer(...)`
- `src/routers/agentic_ask.py`
  - calls `evaluation_service.evaluate_answer(...)`
- `src/schemas/api/ask.py`
  - returns `evaluation` in API responses

The offline evaluation pipeline is also already present:

- `scripts/generate_eval_dataset.py`
  - builds a synthetic JSONL evaluation set from OpenSearch, PostgreSQL, and Neo4j
- `scripts/run_eval.py`
  - runs the standard RAG pipeline on each row and stores evaluation results

## 2. Required packages

Make sure these packages are installed. They already exist in `pyproject.toml`:

```toml
ragas>=0.2.15
datasets>=4.0.0
```

If you sync dependencies with `uv`, use:

```bash
uv sync
```

## 3. Required environment variables

Evaluation behavior is controlled by `EvaluationSettings` in `src/config.py`.

Add or confirm these values in `.env`:

```env
# Evaluation Configuration
EVALUATION__ENABLED=false
EVALUATION__RUN_RAGAS=true
EVALUATION__RUN_LLM_JUDGE=true
EVALUATION__JUDGE_PROVIDER=nvidia
EVALUATION__JUDGE_MODEL=nemotron-3-nano-30b-a3bs
EVALUATION__REQUIRE_REFERENCE=false
EVALUATION__MAX_CONTEXTS=5
```

If you want the judge to use NVIDIA instead of Ollama:

```env
EVALUATION__JUDGE_PROVIDER=nvidia
EVALUATION__JUDGE_MODEL=nvidia/nemotron-3-nano-30b-a3b
NVIDIA_API_KEY=...
```

## 4. Runtime integration in the API

The live API integration should follow this flow:

1. Retrieve chunks from OpenSearch.
2. Generate the answer from the LLM.
3. Pass `query`, `answer`, retrieved `contexts`, and optional `reference_answer` into `evaluation_service.evaluate_answer(...)`.
4. Return the `EvaluationSummary` inside the API response.

This is already implemented in:

- `src/routers/ask.py`
- `src/routers/agentic_ask.py`

The exact call shape is:

```python
evaluation = await evaluation_service.evaluate_answer(
    query=request.query,
    answer=answer,
    contexts=[chunk.get("chunk_text", "") for chunk in chunks],
    reference_answer=reference_answer,  # optional
    metadata={
        "endpoint": "ask",
        "model": request.model,
        "search_mode": "hybrid",
    },
)
```

To integrate the same evaluation logic into another endpoint or service, reuse this exact pattern.

## 5. How RAGAS is integrated

`RAGAS` is executed inside `src/services/evaluation/service.py` in `_run_ragas(...)`.

Current behavior:

- builds a single-row `datasets.Dataset`
- evaluates with:
  - `faithfulness`
  - `answer_relevancy`
  - `context_precision`
- adds `answer_correctness` when `reference_answer` exists

Input mapping used by the code:

- `question` -> user query
- `answer` -> generated answer
- `contexts` -> retrieved chunk texts
- `ground_truth` -> optional reference answer

Necessary rule:

- If you want correctness-style metrics, your dataset or API call must provide `reference_answer`.

## 6. How LLM-as-judge is integrated

`LLM-as-judge` is executed inside `src/services/evaluation/service.py` in `_run_llm_judge(...)`.

Current behavior:

- builds a grading prompt with:
  - question
  - generated answer
  - optional reference answer
  - retrieved contexts
- calls the selected judge model
- expects strict JSON:

```json
{
  "score": 0.0,
  "verdict": "string",
  "reasoning": "string"
}
```

Necessary rule:

- keep `temperature=0`
- keep JSON output enforced
- do not grade only against the answer text; always include retrieved contexts so hallucinations can be penalized

## 7. Offline evaluation workflow

Use offline evaluation for repeatable benchmarking.

### Step 1: generate a dataset

```bash
python -m scripts.generate_eval_dataset --output data/eval/synthetic_eval_dataset.jsonl --size 50
```

The generator now follows this order:

1. sample source contexts from OpenSearch, PostgreSQL, and Neo4j
2. convert them to LangChain `Document` objects
3. use `ragas.testset.TestsetGenerator.generate_with_langchain_docs(...)` when available
4. request a mixed query distribution:
   - `simple` -> `SingleHopSpecificQuerySynthesizer`
   - `reasoning` -> `MultiHopSpecificQuerySynthesizer`
   - `multi_context` -> `MultiHopAbstractQuerySynthesizer`
5. map the RAGAS output back into the project's JSONL schema

If RAGAS testset generation cannot be used, the code falls back to the existing prompt-based synthetic generator.

Each JSONL row should contain:

```json
{
  "question": "...",
  "ground_truth": "...",
  "question_type": "simple",
  "difficulty": "easy",
  "contexts": ["..."],
  "source_kind": "opensearch",
  "arxiv_ids": ["1234.5678"],
  "metadata": {}
}
```

### Step 2: run the evaluator

```bash
python -m scripts.run_eval --dataset data/eval/synthetic_eval_dataset.jsonl --output data/eval/offline_eval_results.jsonl --top-k 3
```

The runner:

1. retrieves documents for each question
2. generates a RAG answer
3. calls `evaluate_answer(...)`
4. stores both the generated answer and evaluation summary

## 8. What you still need to do next

If your goal is to continue implementation after finishing tests, the next concrete steps are:

1. Standardize on the new service only.
   - Keep using `src/services/evaluation/service.py`.
   - Stop adding features to `src/services/evaluator.py`.

2. Turn evaluation on in `.env`.
   - Without `EVALUATION__ENABLED=true`, API responses will not include scores.

3. Create a stable evaluation dataset.
   - Start with synthetic JSONL from `scripts/generate_eval_dataset.py`.
   - Then add a smaller hand-curated dataset for important product queries.

4. Run offline benchmarking before changing retrieval or prompting.
   - Use `scripts/run_eval.py` as the baseline runner.

5. Add reference answers where possible.
   - This unlocks stronger metrics such as `answer_correctness`.

6. Track results over time.
   - Save JSONL outputs from each experiment run.
   - Compare average `faithfulness`, `answer_relevancy`, judge score, and failure cases.

7. Add API tests that assert evaluation wiring.
   - Mock `EvaluationDep`.
   - Verify `evaluation` appears in `/api/v1/ask` and `/api/v1/ask-agentic` responses when enabled.

## 9. Recommended code cleanup

To avoid confusion, do this cleanup next:

1. Deprecate or remove `src/services/evaluator.py`.
2. Deprecate or remove `src/routers/evaluate.py`.
3. Replace the legacy evaluator test with tests that target:
   - `src/services/evaluation/service.py`
   - `src/services/evaluation/offline.py`

Tests should target the canonical service under `src/services/evaluation/`, not the removed legacy evaluator.

## 10. Minimal integration checklist

Use this checklist when adding evaluation to any new RAG flow:

1. Ensure the flow returns retrieved chunk texts.
2. Ensure the flow returns the final generated answer.
3. Inject `EvaluationService`.
4. Call `evaluate_answer(query, answer, contexts, reference_answer, metadata)`.
5. Return `EvaluationSummary` in the response schema.
6. Add one unit test for success and one for graceful failure.

## 11. Suggested next task order

If you want to continue implementation in a clean sequence, do it in this order:

1. Remove legacy evaluation path.
2. Add unit tests for `src/services/evaluation/service.py`.
3. Add API tests for evaluation presence in `/ask` and `/ask-agentic`.
4. Generate a baseline dataset.
5. Run a baseline offline benchmark.
6. Improve retrieval or prompting based on the worst-scoring examples.

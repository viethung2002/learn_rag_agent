import argparse
import asyncio
import logging
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings
from src.services.embeddings.factory import make_embeddings_service
from src.services.evaluation.factory import make_evaluation_service
from src.services.evaluation.offline import StandardRAGEvaluatorRunner, close_async_resources
from src.services.nvidia.client import NvidiaClient
from src.services.opensearch.factory import make_opensearch_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline RAG evaluation on a JSONL dataset.")
    parser.add_argument("--dataset", default="data/eval/synthetic_eval_dataset.jsonl", help="Input JSONL dataset path.")
    parser.add_argument("--output", default="data/eval/offline_eval_results.jsonl", help="Output JSONL results path.")
    parser.add_argument("--model", default=None, help="Generator model for the standard RAG answerer.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of rows to evaluate.")
    parser.add_argument("--bm25-only", action="store_true", help="Disable hybrid retrieval.")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    settings = get_settings()
    answer_model = args.model or settings.nvidia_model
    use_nvidia = bool(answer_model and str(answer_model).startswith("nvidia/"))

    nvidia_client = None
    if use_nvidia:
        env_api_key = os.environ.get("NVIDIA_API_KEY")
        if env_api_key:
            try:
                settings = settings.model_copy(update={"nvidia_api_key": env_api_key})
            except Exception:
                pass
        nvidia_client = NvidiaClient(settings)
    opensearch_client = make_opensearch_client()
    embeddings_client = make_embeddings_service()
    evaluation_service = make_evaluation_service(
        settings=settings,
        nvidia_client=nvidia_client,
        langfuse_tracer=None,
    )

    evaluation_overrides = {"enabled": True, "require_reference": False}
    if use_nvidia:
        evaluation_overrides.update(
            {
                "judge_provider": "nvidia",
                "judge_model": answer_model,
            }
        )

    evaluation_service.config = evaluation_service.config.model_copy(
        update=evaluation_overrides
    )

    try:
        runner = StandardRAGEvaluatorRunner(
            opensearch_client=opensearch_client,
            embeddings_client=embeddings_client,
            evaluation_service=evaluation_service,
            nvidia_client=nvidia_client,
        )
        await runner.run(
            dataset_path=Path(args.dataset),
            output_path=Path(args.output),
            model=answer_model,
            top_k=args.top_k,
            use_hybrid=not args.bm25_only,
            limit=args.limit,
        )
    finally:
        await close_async_resources(embeddings_client, nvidia_client)


if __name__ == "__main__":
    asyncio.run(main())

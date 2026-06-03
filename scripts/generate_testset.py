"""Backward-compatible shim for the new evaluation dataset generator."""

import asyncio
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings
from src.db.factory import make_database
from src.services.evaluation.offline import SyntheticDatasetGenerator, close_async_resources
from src.services.neo4j.factory import make_neo4j_client
from src.services.ollama.factory import make_ollama_client
from src.services.opensearch.factory import make_opensearch_client


async def main(out: str = "data/eval/synthetic_eval_dataset.jsonl", size: int = 50) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    settings = get_settings()
    database = None
    neo4j_client = None
    ollama_client = None
    opensearch_client = make_opensearch_client()

    try:
        try:
            database = make_database()
        except Exception as exc:
            logging.warning("PostgreSQL unavailable, PostgreSQL-based sampling will be skipped: %s", exc)

        ollama_client = make_ollama_client()

        try:
            neo4j_client = make_neo4j_client(settings)
            neo4j_client.verify_connectivity()
        except Exception:
            neo4j_client = None

        session_manager = database.get_session() if database is not None else None
        if session_manager is None:
            generator = SyntheticDatasetGenerator(
                opensearch_client=opensearch_client,
                session=None,
                generator_provider="ollama",
                generator_model=settings.ollama_model,
                ollama_client=ollama_client,
                neo4j_client=neo4j_client,
            )
            await generator.generate_dataset(output_path=Path(out), total_size=size)
        else:
            with session_manager as session:
                generator = SyntheticDatasetGenerator(
                    opensearch_client=opensearch_client,
                    session=session,
                    generator_provider="ollama",
                    generator_model=settings.ollama_model,
                    ollama_client=ollama_client,
                    neo4j_client=neo4j_client,
                )
                await generator.generate_dataset(output_path=Path(out), total_size=size)
    finally:
        if database is not None:
            database.teardown()
        if neo4j_client is not None:
            neo4j_client.close()
        await close_async_resources(ollama_client)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Deprecated wrapper. Use `python -m scripts.generate_eval_dataset` instead."
    )
    parser.add_argument("--out", default="data/eval/synthetic_eval_dataset.jsonl")
    parser.add_argument("--size", type=int, default=50)
    args = parser.parse_args()

    asyncio.run(main(out=args.out, size=args.size))

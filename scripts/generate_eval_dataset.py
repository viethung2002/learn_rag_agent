import argparse
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
from src.services.nvidia.factory import make_nvidia_client
from src.services.ollama.factory import make_ollama_client
from src.services.opensearch.factory import make_opensearch_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic RAG evaluation dataset from existing project data.")
    parser.add_argument("--output", default="data/eval/synthetic_eval_dataset.jsonl", help="Output JSONL path.")
    parser.add_argument("--size", type=int, default=50, help="Number of examples to generate.")
    parser.add_argument(
        "--generator-provider",
        choices=["ollama", "nvidia"],
        default="ollama",
        help="LLM provider used to synthesize Q/A pairs.",
    )
    parser.add_argument("--generator-model", default=None, help="Override model used to synthesize Q/A pairs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--opensearch-weight", type=float, default=0.5, help="Sampling weight for OpenSearch examples.")
    parser.add_argument("--postgres-weight", type=float, default=0.3, help="Sampling weight for PostgreSQL examples.")
    parser.add_argument("--neo4j-weight", type=float, default=0.2, help="Sampling weight for Neo4j examples.")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    settings = get_settings()
    generator_model = args.generator_model or (
        settings.ollama_model if args.generator_provider == "ollama" else settings.nvidia_model
    )

    database = None
    opensearch_client = make_opensearch_client()
    neo4j_client = None
    ollama_client = None
    nvidia_client = None

    try:
        try:
            database = make_database()
        except Exception as exc:
            logging.warning("PostgreSQL unavailable, PostgreSQL-based sampling will be skipped: %s", exc)

        if args.generator_provider == "ollama":
            ollama_client = make_ollama_client()
        else:
            nvidia_client = make_nvidia_client()

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
                generator_provider=args.generator_provider,
                generator_model=generator_model,
                ollama_client=ollama_client,
                nvidia_client=nvidia_client,
                neo4j_client=neo4j_client,
                seed=args.seed,
            )
            await generator.generate_dataset(
                output_path=Path(args.output),
                total_size=args.size,
                source_distribution={
                    "opensearch": args.opensearch_weight,
                    "postgres": args.postgres_weight,
                    "neo4j": args.neo4j_weight,
                },
            )
        else:
            with session_manager as session:
                generator = SyntheticDatasetGenerator(
                    opensearch_client=opensearch_client,
                    session=session,
                    generator_provider=args.generator_provider,
                    generator_model=generator_model,
                    ollama_client=ollama_client,
                    nvidia_client=nvidia_client,
                    neo4j_client=neo4j_client,
                    seed=args.seed,
                )
                await generator.generate_dataset(
                    output_path=Path(args.output),
                    total_size=args.size,
                    source_distribution={
                        "opensearch": args.opensearch_weight,
                        "postgres": args.postgres_weight,
                        "neo4j": args.neo4j_weight,
                    },
                )
    finally:
        if database is not None:
            database.teardown()
        if neo4j_client is not None:
            neo4j_client.close()
        await close_async_resources(ollama_client, nvidia_client)


if __name__ == "__main__":
    asyncio.run(main())

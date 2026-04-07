import asyncio
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional

from sqlalchemy.orm import Session

from src.models.paper import Paper
from src.repositories.paper import PaperRepository
from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.evaluation.service import EvaluationService
from src.services.neo4j.client import Neo4jClient
from src.services.nvidia.client import NvidiaClient
from src.services.opensearch.client import OpenSearchClient

logger = logging.getLogger(__name__)

SourceKind = Literal["opensearch", "postgres", "neo4j"]


@dataclass
class SourceSample:
    source_kind: SourceKind
    arxiv_ids: List[str]
    contexts: List[str]
    metadata: Dict[str, Any]


RAGAS_QUERY_DISTRIBUTION: Dict[str, float] = {
    "simple": 0.4,
    "reasoning": 0.3,
    "multi_context": 0.3,
}


class SyntheticDatasetGenerator:
    """Generate evaluation datasets from the project's existing stores using NVIDIA only."""

    def __init__(
        self,
        opensearch_client: OpenSearchClient,
        session: Session | None,
        generator_provider: Literal["nvidia"],
        generator_model: str,
        nvidia_client: NvidiaClient | None = None,
        neo4j_client: Neo4jClient | None = None,
        seed: int = 42,
    ) -> None:
        self.opensearch = opensearch_client
        self.session = session
        self.paper_repo = PaperRepository(session) if session is not None else None
        self.generator_provider = generator_provider
        self.generator_model = generator_model
        self.nvidia_client = nvidia_client
        self.neo4j_client = neo4j_client
        self.random = random.Random(seed)

    async def generate_dataset(
        self,
        output_path: str | Path,
        total_size: int = 50,
        source_distribution: Optional[Dict[SourceKind, float]] = None,
        max_context_chars: int = 2200,
    ) -> List[Dict[str, Any]]:
        distribution = source_distribution or {"opensearch": 0.5, "postgres": 0.3, "neo4j": 0.2}
        plan = self._build_generation_plan(total_size, distribution)
        sampled_sources: List[SourceSample] = []

        for source_kind in plan:
            sample = self._sample_source(source_kind, max_context_chars=max_context_chars)
            if sample is None:
                logger.warning("Skipping sample because source '%s' returned nothing", source_kind)
                continue
            sampled_sources.append(sample)

        rows = await self._generate_dataset_with_ragas(samples=sampled_sources, total_size=total_size)
        if not rows:
            rows = []
            for sample in sampled_sources:
                row = await self._synthesize_example(sample)
                if row:
                    rows.append(row)

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        logger.info("Generated %s synthetic evaluation rows at %s", len(rows), output)
        return rows

    async def _generate_dataset_with_ragas(self, samples: List[SourceSample], total_size: int) -> List[Dict[str, Any]]:
        if not samples:
            return []

        try:
            from langchain_core.documents import Document as LCDocument
            from ragas.testset import TestsetGenerator
            from ragas.testset.synthesizers import (
                MultiHopAbstractQuerySynthesizer,
                MultiHopSpecificQuerySynthesizer,
                SingleHopSpecificQuerySynthesizer,
            )
        except ImportError:
            logger.info("RAGAS testset generation dependencies are unavailable; using prompt-based synthetic generation")
            return []
        except Exception as exc:
            logger.warning("Failed to import RAGAS testset generation stack: %s", exc)
            return []

        langchain_llm = self._make_langchain_llm()
        langchain_embeddings = self._make_langchain_embeddings()
        if langchain_llm is None or langchain_embeddings is None:
            logger.info("RAGAS testset generation skipped because no compatible LLM/embeddings adapter is available")
            return []

        documents: List[LCDocument] = []
        context_lookup: Dict[str, Dict[str, Any]] = {}
        for sample_index, sample in enumerate(samples):
            for context_index, context in enumerate(sample.contexts):
                if not context.strip():
                    continue
                metadata = {
                    "sample_id": sample_index,
                    "context_index": context_index,
                    "source_kind": sample.source_kind,
                    "arxiv_ids": sample.arxiv_ids,
                    "sample_metadata": sample.metadata,
                }
                documents.append(LCDocument(page_content=context, metadata=metadata))
                context_lookup[context] = metadata

        if not documents:
            return []

        try:
            generator = TestsetGenerator.from_langchain(
                llm=langchain_llm,
                embedding_model=langchain_embeddings,
                llm_context=(
                    "Generate evaluation questions for an arXiv paper RAG system. "
                    "Include a balanced mix of direct factual questions, reasoning questions, "
                    "and multi-context questions that require combining information across contexts."
                ),
            )
            query_distribution = [
                (SingleHopSpecificQuerySynthesizer(llm=generator.llm), RAGAS_QUERY_DISTRIBUTION["simple"]),
                (MultiHopSpecificQuerySynthesizer(llm=generator.llm), RAGAS_QUERY_DISTRIBUTION["reasoning"]),
                (MultiHopAbstractQuerySynthesizer(llm=generator.llm), RAGAS_QUERY_DISTRIBUTION["multi_context"]),
            ]

            testset = generator.generate_with_langchain_docs(
                documents=documents,
                testset_size=total_size,
                query_distribution=query_distribution,
                with_debugging_logs=False,
                raise_exceptions=False,
            )
        except Exception as exc:
            logger.warning("RAGAS testset generation failed, falling back to prompt-based generation: %s", exc)
            return []

        rows: List[Dict[str, Any]] = []
        for row in testset.to_list():
            mapped_row = self._convert_ragas_row(row=row, context_lookup=context_lookup)
            if mapped_row is not None:
                rows.append(mapped_row)
        return rows

    def _build_generation_plan(self, total_size: int, distribution: Dict[SourceKind, float]) -> List[SourceKind]:
        normalized_total = sum(distribution.values()) or 1.0
        plan: List[SourceKind] = []
        for source_kind, weight in distribution.items():
            count = round(total_size * (weight / normalized_total))
            plan.extend([source_kind] * count)

        while len(plan) < total_size:
            plan.append("opensearch")

        self.random.shuffle(plan)
        return plan[:total_size]

    def _sample_source(self, source_kind: SourceKind, max_context_chars: int) -> SourceSample | None:
        if source_kind == "opensearch":
            return self._sample_opensearch(max_context_chars=max_context_chars)
        if source_kind == "postgres":
            return self._sample_postgres(max_context_chars=max_context_chars)
        if source_kind == "neo4j":
            return self._sample_neo4j(max_context_chars=max_context_chars)
        return None

    def _sample_opensearch(self, max_context_chars: int) -> SourceSample | None:
        try:
            response = self.opensearch.client.search(
                index=self.opensearch.index_name,
                body={
                    "size": 1,
                    "query": {"function_score": {"query": {"match_all": {}}, "random_score": {}}},
                    "_source": {
                        "includes": [
                            "arxiv_id",
                            "chunk_text",
                            "title",
                            "authors",
                            "categories",
                            "section_name",
                        ]
                    },
                },
            )
            hits = response.get("hits", {}).get("hits", [])
            if not hits:
                return None

            source = hits[0]["_source"]
            context = (source.get("chunk_text") or "")[:max_context_chars]
            if not context.strip():
                return None

            return SourceSample(
                source_kind="opensearch",
                arxiv_ids=[source.get("arxiv_id", "")],
                contexts=[context],
                metadata={
                    "title": source.get("title"),
                    "authors": source.get("authors"),
                    "categories": source.get("categories"),
                    "section_name": source.get("section_name"),
                },
            )
        except Exception as exc:
            logger.warning("OpenSearch sampling failed: %s", exc)
            return None

    def _sample_postgres(self, max_context_chars: int) -> SourceSample | None:
        if self.paper_repo is None:
            return None
        papers = self.paper_repo.get_papers_with_raw_text(limit=30)
        if not papers:
            papers = self.paper_repo.get_processed_papers(limit=30)
        if not papers:
            papers = self.paper_repo.get_all(limit=30)
        if not papers:
            return None

        paper = self.random.choice(papers)
        contexts = self._paper_to_contexts(paper, max_context_chars=max_context_chars)
        if not contexts:
            return None

        return SourceSample(
            source_kind="postgres",
            arxiv_ids=[paper.arxiv_id],
            contexts=contexts,
            metadata={
                "title": paper.title,
                "authors": paper.authors,
                "categories": paper.categories,
                "published_date": paper.published_date.isoformat() if paper.published_date else None,
            },
        )

    def _sample_neo4j(self, max_context_chars: int) -> SourceSample | None:
        if not self.neo4j_client:
            return None

        queries = [
            (
                """
                MATCH (p1:Paper)-[:CITES_PAPER]->(p2:Paper)
                WHERE p1.abstract IS NOT NULL AND p2.abstract IS NOT NULL
                RETURN p1.arxiv_id AS arxiv_id_1, p1.title AS title_1, p1.abstract AS abstract_1,
                       p2.arxiv_id AS arxiv_id_2, p2.title AS title_2, p2.abstract AS abstract_2
                LIMIT 25
                """,
                "citation_pair",
            ),
            (
                """
                MATCH (a:Author)-[:WROTE]->(p1:Paper)
                MATCH (a)-[:WROTE]->(p2:Paper)
                WHERE p1.arxiv_id <> p2.arxiv_id
                  AND p1.abstract IS NOT NULL AND p2.abstract IS NOT NULL
                RETURN p1.arxiv_id AS arxiv_id_1, p1.title AS title_1, p1.abstract AS abstract_1,
                       p2.arxiv_id AS arxiv_id_2, p2.title AS title_2, p2.abstract AS abstract_2,
                       a.name AS author_name
                LIMIT 25
                """,
                "shared_author_pair",
            ),
        ]

        for query, relation_kind in queries:
            try:
                rows = self.neo4j_client.execute_read(query)
                rows = [row for row in rows if row.get("abstract_1") and row.get("abstract_2")]
                if not rows:
                    continue
                row = self.random.choice(rows)
                contexts = [
                    f"Paper 1: {row.get('title_1', '')}\n{row.get('abstract_1', '')[: max_context_chars // 2]}",
                    f"Paper 2: {row.get('title_2', '')}\n{row.get('abstract_2', '')[: max_context_chars // 2]}",
                ]
                return SourceSample(
                    source_kind="neo4j",
                    arxiv_ids=[row.get("arxiv_id_1", ""), row.get("arxiv_id_2", "")],
                    contexts=contexts,
                    metadata={"relation_kind": relation_kind, "author_name": row.get("author_name")},
                )
            except Exception as exc:
                logger.warning("Neo4j sampling query failed: %s", exc)

        return None

    async def _synthesize_example(self, sample: SourceSample) -> Dict[str, Any] | None:
        prompt = self._build_synthesis_prompt(sample)
        schema = {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "ground_truth": {"type": "string"},
                "question_type": {"type": "string"},
                "difficulty": {"type": "string"},
            },
            "required": ["question", "ground_truth", "question_type", "difficulty"],
        }

        raw = await self._generate_structured_json(prompt=prompt, schema=schema)
        if not raw:
            return None

        return {
            "question": raw["question"],
            "ground_truth": raw["ground_truth"],
            "question_type": raw["question_type"],
            "difficulty": raw["difficulty"],
            "contexts": sample.contexts,
            "source_kind": sample.source_kind,
            "arxiv_ids": sample.arxiv_ids,
            "metadata": sample.metadata,
        }

    def _make_langchain_llm(self) -> Any | None:
        if self.nvidia_client:
            return self.nvidia_client.get_langchain_model(model=self.generator_model, temperature=0.2)
        return None

    def _make_langchain_embeddings(self) -> Any | None:
        if self.nvidia_client:
            from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

            return NVIDIAEmbeddings(
                model="nvidia/nv-embedqa-e5-v5",
                nvidia_api_key=self.nvidia_client.api_key,
            )
        return None

    def _convert_ragas_row(
        self,
        row: Dict[str, Any],
        context_lookup: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any] | None:
        question = row.get("user_input")
        reference = row.get("reference")
        contexts = [context for context in row.get("reference_contexts", []) if context]

        if not question or not reference or not contexts:
            return None

        matched = [context_lookup.get(context) for context in contexts]
        matched = [item for item in matched if item is not None]

        source_kind = matched[0]["source_kind"] if matched else "opensearch"
        arxiv_ids = []
        for item in matched:
            arxiv_ids.extend(item.get("arxiv_ids", []))

        merged_metadata: Dict[str, Any] = {
            "persona_name": row.get("persona_name"),
            "query_style": row.get("query_style"),
            "query_length": row.get("query_length"),
            "synthesizer_name": row.get("synthesizer_name"),
        }
        for item in matched:
            sample_metadata = item.get("sample_metadata", {})
            for key, value in sample_metadata.items():
                merged_metadata.setdefault(key, value)

        question_type = self._map_ragas_question_type(str(row.get("synthesizer_name") or ""))
        return {
            "question": question,
            "ground_truth": reference,
            "question_type": question_type,
            "difficulty": self._infer_difficulty(question_type),
            "contexts": contexts,
            "source_kind": source_kind,
            "arxiv_ids": list(dict.fromkeys(arxiv_ids)),
            "metadata": merged_metadata,
        }

    @staticmethod
    def _map_ragas_question_type(synthesizer_name: str) -> str:
        if "single_hop" in synthesizer_name:
            return "simple"
        if "abstract" in synthesizer_name:
            return "multi_context"
        return "reasoning"

    @staticmethod
    def _infer_difficulty(question_type: str) -> str:
        if question_type == "simple":
            return "easy"
        if question_type == "reasoning":
            return "medium"
        return "hard"

    def _build_synthesis_prompt(self, sample: SourceSample) -> str:
        context_block = "\n\n".join(
            f"[Context {index + 1}]\n{context}" for index, context in enumerate(sample.contexts)
        )
        style_hint = {
            "opensearch": "Create one natural single-context fact or explanation question.",
            "postgres": "Create one question grounded in the paper metadata/abstract.",
            "neo4j": "Create one multi-context reasoning question that needs both contexts.",
        }[sample.source_kind]

        return (
            "You are creating a high-quality RAG evaluation example.\n"
            "Use only the provided contexts.\n"
            "Do not invent facts outside the contexts.\n"
            "Return valid JSON only.\n"
            f"{style_hint}\n\n"
            "Fields:\n"
            "- question: realistic user question\n"
            "- ground_truth: concise but complete reference answer\n"
            "- question_type: simple, reasoning, or multi_context\n"
            "- difficulty: easy, medium, or hard\n\n"
            f"Arxiv IDs: {sample.arxiv_ids}\n"
            f"Metadata: {json.dumps(sample.metadata, ensure_ascii=False)}\n\n"
            f"{context_block}\n"
        )

    async def _generate_structured_json(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any] | None:
        try:
            if not self.nvidia_client:
                return None
            response = await self.nvidia_client.generate(
                model=self.generator_model,
                prompt=prompt,
                temperature=0.2,
                top_p=0.9,
                format=schema,
            )
            # Nvidia client may return parsed object or text
            if isinstance(response, dict) and response.get("parsed") is not None:
                return response.get("parsed")
            raw_text = (response or {}).get("text")
            if not raw_text:
                return None
            return json.loads(raw_text)
        except Exception as exc:
            logger.warning("Synthetic example generation failed: %s", exc)
            return None

    @staticmethod
    def _paper_to_contexts(paper: Paper, max_context_chars: int) -> List[str]:
        contexts: List[str] = []
        if paper.abstract:
            contexts.append(f"Title: {paper.title}\nAbstract: {paper.abstract[:max_context_chars]}")

        if paper.raw_text:
            contexts.append(paper.raw_text[:max_context_chars])

        return contexts[:2]


class StandardRAGEvaluatorRunner:
    """Run the existing standard RAG pipeline against a JSONL dataset using Nvidia only."""

    def __init__(
        self,
        opensearch_client: OpenSearchClient,
        embeddings_client: JinaEmbeddingsClient,
        evaluation_service: EvaluationService,
        nvidia_client: NvidiaClient | None = None,
    ) -> None:
        self.opensearch = opensearch_client
        self.embeddings = embeddings_client
        self.nvidia = nvidia_client
        self.evaluation_service = evaluation_service

    async def run(
        self,
        dataset_path: str | Path,
        output_path: str | Path,
        model: str,
        top_k: int = 3,
        use_hybrid: bool = True,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        dataset_rows = list(self._load_jsonl(dataset_path))
        if limit is not None:
            dataset_rows = dataset_rows[:limit]

        results: List[Dict[str, Any]] = []
        for index, row in enumerate(dataset_rows, start=1):
            logger.info("Evaluating row %s/%s", index, len(dataset_rows))
            result = await self._run_single(
                question=row["question"],
                reference_answer=row.get("ground_truth"),
                source_row=row,
                model=model,
                top_k=top_k,
                use_hybrid=use_hybrid,
            )
            results.append(result)

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")

        logger.info("Saved %s evaluation results to %s", len(results), output)
        return results

    async def _run_single(
        self,
        question: str,
        reference_answer: Optional[str],
        source_row: Dict[str, Any],
        model: str,
        top_k: int,
        use_hybrid: bool,
    ) -> Dict[str, Any]:
        query_embedding = None
        if use_hybrid:
            query_embedding = await self.embeddings.embed_query(question)

        search_results = self.opensearch.search_unified(
            query=question,
            query_embedding=query_embedding,
            size=top_k,
            use_hybrid=use_hybrid and query_embedding is not None,
        )
        hits = search_results.get("hits", [])
        chunks = [
            {
                "arxiv_id": hit.get("arxiv_id"),
                "chunk_text": hit.get("chunk_text", hit.get("abstract", "")),
            }
            for hit in hits
        ]

        if not self.nvidia:
            raise RuntimeError("Nvidia client is required for evaluation but not configured")

        try:
            rag_response = await self.nvidia.generate_rag_answer(
                query=question,
                chunks=chunks,
                model=model,
            )
        except Exception as exc:
            logger.error("RAG generation failed for model=%s: %s", model, exc)
            rag_response = {}

        answer = rag_response.get("answer") or rag_response.get("response") or rag_response.get("text") or ""

        # Force judge to use Nvidia as well
        evaluation = await self.evaluation_service.evaluate_answer(
            query=question,
            answer=answer,
            contexts=[chunk["chunk_text"] for chunk in chunks if chunk["chunk_text"]],
            reference_answer=reference_answer,
            metadata={
                "mode": "offline_standard_rag",
                "top_k": top_k,
                "use_hybrid": use_hybrid,
                "dataset_source_kind": source_row.get("source_kind"),
            },
            judge_provider_override="nvidia",
            judge_model_override=model,
        )

        return {
            "question": question,
            "ground_truth": reference_answer,
            "generated_answer": answer,
            "retrieved_contexts": [chunk["chunk_text"] for chunk in chunks if chunk["chunk_text"]],
            "retrieved_arxiv_ids": [chunk["arxiv_id"] for chunk in chunks if chunk.get("arxiv_id")],
            "source_kind": source_row.get("source_kind"),
            "source_arxiv_ids": source_row.get("arxiv_ids", []),
            "evaluation": evaluation.model_dump() if evaluation else None,
        }

    @staticmethod
    def _load_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)


async def close_async_resources(*resources: Any) -> None:
    """Close async-capable resources without failing the calling script."""
    for resource in resources:
        if resource is None:
            continue
        close_method = getattr(resource, "close", None)
        if close_method is None:
            continue
        try:
            maybe_awaitable = close_method()
            if asyncio.iscoroutine(maybe_awaitable):
                await maybe_awaitable
        except Exception as exc:
            logger.warning("Failed to close resource cleanly: %s", exc)


    

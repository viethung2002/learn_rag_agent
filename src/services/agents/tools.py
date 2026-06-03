import asyncio
import logging
from typing import Any, Dict, List

from langchain_core.documents import Document

from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.opensearch.client import OpenSearchClient
from src.services.neo4j.client import Neo4jClient

logger = logging.getLogger(__name__)


def create_retriever_tool(
    opensearch_client: OpenSearchClient,
    embeddings_client: JinaEmbeddingsClient,
    top_k: int = 3,
    use_hybrid: bool = True,
):
    """Create a retriever tool that wraps OpenSearch service.

    :param opensearch_client: Existing OpenSearch service
    :param embeddings_client: Existing Jina embeddings service
    :param top_k: Number of chunks to retrieve
    :param use_hybrid: Use hybrid search (BM25 + vector)
    :returns: LangChain tool for retrieving papers
    """

    async def retrieve_papers(query: str) -> list[Document]:
        """Search and return relevant arXiv research papers.

        Use this tool when the user asks about:
        - Machine learning concepts or techniques
        - Deep learning architectures
        - Natural language processing
        - Computer vision methods
        - AI research topics
        - Specific algorithms or models

        :param query: The search query describing what papers to find
        :returns: List of relevant paper excerpts with metadata
        """
        logger.info(f"Retrieving papers for query: {query[:100]}...")
        logger.debug(f"Search mode: {'hybrid' if use_hybrid else 'bm25'}, top_k: {top_k}")

        # Generate query embedding
        logger.debug("Generating query embedding")
        query_embedding = await embeddings_client.embed_query(query)
        logger.debug(f"Generated embedding with {len(query_embedding)} dimensions")

        # Search using OpenSearch
        logger.debug("Searching OpenSearch")
        loop = asyncio.get_running_loop()
        search_results = await loop.run_in_executor(
            None,
            lambda: opensearch_client.search_unified(
                query=query,
                query_embedding=query_embedding,
                size=top_k,
                use_hybrid=use_hybrid,
            ),
        )

        # Convert SearchHit to LangChain Document
        documents: List[Document] = []
        hits = search_results.get("hits", [])
        logger.info(f"Found {len(hits)} documents from OpenSearch")

        # deduplicate hits by arxiv_id while preserving order
        seen: set = set()
        unique_hits = []
        for hit in hits:
            aid = hit.get("arxiv_id")
            if not aid:
                unique_hits.append(hit)
                continue
            if aid in seen:
                continue
            seen.add(aid)
            unique_hits.append(hit)

        for hit in unique_hits:
            doc = Document(
                page_content=hit["chunk_text"],
                metadata={
                    "arxiv_id": hit["arxiv_id"],
                    "title": hit.get("title", ""),
                    "authors": hit.get("authors", ""),
                    "score": hit.get("score", 0.0),
                    "source": f"https://arxiv.org/pdf/{hit['arxiv_id']}.pdf",
                    "section": hit.get("section_name", ""),
                    "search_mode": "hybrid" if use_hybrid else "bm25",
                    "top_k": top_k,
                },
            )
            documents.append(doc)

        logger.debug(f"Converted {len(documents)} hits to LangChain Documents")
        logger.info(f"✓ Retrieved {len(documents)} papers successfully")

        return documents

    return retrieve_papers


def create_citation_tool(neo4j_client: Neo4jClient):
    """Create a tool that returns shared citations between two arXiv papers.

    The returned tool expects two arguments: `arxiv_id_a` and `arxiv_id_b`.
    It executes a Cypher query (in an executor) to find reference nodes cited by
    both papers and returns a list of summaries (title, arxiv_id, labels).
    """

    async def shared_citations(arxiv_id_a: str, arxiv_id_b: str) -> List[Dict[str, Any]]:
        """Return shared citations between two papers identified by arXiv IDs."""
        logger.info(f"Finding shared citations for {arxiv_id_a} and {arxiv_id_b}")
        cypher = (
            "MATCH (a {arxiv_id:$a}), (b {arxiv_id:$b})\n"
            "MATCH (a)-[:CITES|CITES_PAPER]->(r)<-[:CITES|CITES_PAPER]-(b)\n"
            "RETURN DISTINCT r.title AS title, r.arxiv_id AS arxiv_id, labels(r) AS labels LIMIT 200"
        )
        params = {"a": arxiv_id_a, "b": arxiv_id_b}

        loop = asyncio.get_running_loop()

        def run_query():
            return neo4j_client.execute_read(cypher, params)

        try:
            results = await loop.run_in_executor(None, run_query)
            logger.info(f"Found {len(results)} shared citations")
            return results
        except Exception:
            logger.exception("Error while querying Neo4j for shared citations")
            return []

    return shared_citations


def create_paper_lookup_tool(opensearch_client: OpenSearchClient):
    """Create tools to fetch papers by arXiv id or by exact title.

    - `fetch_papers_by_ids(arxiv_ids: List[str])` returns chunks for each paper.
    - `fetch_paper_by_title(title: str)` returns top BM25 paper matching the title.
    """

    async def fetch_papers_by_ids(arxiv_ids: List[str]) -> List[Dict[str, Any]]:
        """Return chunks/metadata for each arXiv id in `arxiv_ids`."""
        logger.info(f"Fetching {len(arxiv_ids)} papers by arxiv_id")
        loop = asyncio.get_running_loop()

        def run():
            results = []
            for aid in arxiv_ids:
                chunks = opensearch_client.get_chunks_by_paper(aid)
                # attach arxiv_id and title if available
                if chunks:
                    meta = {"arxiv_id": aid, "title": chunks[0].get("title", ""), "chunks": chunks}
                else:
                    meta = {"arxiv_id": aid, "title": "", "chunks": []}
                results.append(meta)
            return results

        return await loop.run_in_executor(None, run)

    async def fetch_paper_by_title(title: str) -> Dict[str, Any]:
        """Search BM25 for a paper by title and return its arxiv_id and top chunks."""
        logger.info(f"Searching for paper by title: {title[:120]}")
        loop = asyncio.get_running_loop()

        def run():
            res = opensearch_client.search_papers(query=title, size=1)
            hits = res.get("hits", [])
            if not hits:
                return {}
            top = hits[0]
            aid = top.get("arxiv_id")
            chunks = opensearch_client.get_chunks_by_paper(aid) if aid else []
            return {"arxiv_id": aid, "title": top.get("title", ""), "chunks": chunks}

        return await loop.run_in_executor(None, run)

    return fetch_papers_by_ids, fetch_paper_by_title

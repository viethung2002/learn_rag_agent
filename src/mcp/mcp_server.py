import asyncio
import logging
from typing import Any, Dict, List, Optional, TypedDict

from fastmcp import FastMCP
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_tavily import TavilySearch

from src.services.embeddings.factory import make_embeddings_client
from src.services.opensearch.factory import make_opensearch_client
from src.services.neo4j.factory import make_neo4j_client

mcp = FastMCP("arxiv-tools")
logger = logging.getLogger(__name__)

@mcp.tool()
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
    opensearch_client = make_opensearch_client()
    embeddings_client = make_embeddings_client()
    top_k = 4
    use_hybrid = True
    logger.info(f"Retrieving papers for query: {query[:100]}...")
    logger.debug(f"Search mode: {'hybrid' if use_hybrid else 'bm25'}, top_k: {top_k}")

    # Generate query embedding
    logger.debug("Generating query embedding")
    query_embedding = await embeddings_client.embed_query(query)
    logger.debug(f"Generated embedding with {len(query_embedding)} dimensions")

    # Search using OpenSearch (run blocking opensearch call in executor)
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
    documents = []
    hits = search_results.get("hits", [])
    logger.info(f"Found {len(hits)} documents from OpenSearch")

    # deduplicate hits by arxiv_id while preserving order
    seen = set()
    unique_hits = []
    for hit in hits:
        aid = hit.get("arxiv_id")
        if not aid or aid not in seen:
            if aid:
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


@mcp.tool()
async def shared_citations(arxiv_id_a: str, arxiv_id_b: str) -> List[Dict[str, Any]]:
    """Return shared citations between two papers (via Neo4j)."""
    neo4j_client = make_neo4j_client()
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


@mcp.tool()
async def fetch_papers_by_ids(arxiv_ids: List[str]) -> List[Dict[str, Any]]:
    """Return chunks/metadata for each arXiv id."""
    opensearch_client = make_opensearch_client()
    logger.info(f"Fetching {len(arxiv_ids)} papers by arxiv_id")
    loop = asyncio.get_running_loop()

    def run():
        results = []
        for aid in arxiv_ids:
            chunks = opensearch_client.get_chunks_by_paper(aid)
            if chunks:
                meta = {"arxiv_id": aid, "title": chunks[0].get("title", ""), "chunks": chunks}
            else:
                meta = {"arxiv_id": aid, "title": "", "chunks": []}
            results.append(meta)
        return results

    return await loop.run_in_executor(None, run)


@mcp.tool()
async def fetch_paper_by_title(title: str) -> Dict[str, Any]:
    """Search BM25 for a paper by title and return its arxiv_id and top chunks."""
    opensearch_client = make_opensearch_client()
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


@mcp.tool()
async def web_search(query: str) -> str:
    """Perform a web search to gather current information.

    Use this tool when the user asks about:
    - Recent events or news
    - Current statistics or data
    - Trending topics
    - Information that may have changed recently

    :param query: The search query describing what information to find
    :returns: A summary of the web search results
    """
    search_tool = TavilySearch(
        max_results=2,
        tavily_api_key="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",  # 👉 nên để ENV thay vì hardcode
    )

    # TavilySearch là Runnable → dùng ainvoke
    results = await search_tool.ainvoke({"query": query})

    # Chuẩn hóa output thành text gọn gàng cho MCP
    if not results or "results" not in results:
        return "No search results found."

    summaries = []
    for r in results["results"]:
        title = r.get("title", "No title")
        content = r.get("content", "")
        url = r.get("url", "")
        summaries.append(f"- {title}\n  {content}\n  Source: {url}")

    return "\n".join(summaries)

if __name__ == "__main__":
    opensearch_client = make_opensearch_client()
    print('Health check:', opensearch_client.health_check())
    mcp.run(
        transport="streamable-http",   # hoặc "http" nếu version mới dùng tên này
        host="0.0.0.0",                # Quan trọng: bind tất cả interface (để Docker truy cập)
        port=8100,                     # Phải khớp với EXPOSE trong Dockerfile
        path="/mcp"                    # Đảm bảo endpoint là /mcp
    )

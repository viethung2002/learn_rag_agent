import logging

from fastmcp import FastMCP
from langchain_core.documents import Document

from src.services.embeddings.factory import make_embeddings_client
from src.services.opensearch.factory import make_opensearch_client

mcp = FastMCP("arxiv-tools")
logger = logging.getLogger(__name__)

opensearch_client = make_opensearch_client()
embeddings_client = make_embeddings_client()
top_k = 4
use_hybrid = True

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
    logger.info(f"Retrieving papers for query: {query[:100]}...")
    logger.debug(f"Search mode: {'hybrid' if use_hybrid else 'bm25'}, top_k: {top_k}")

    # Generate query embedding
    logger.debug("Generating query embedding")
    query_embedding = await embeddings_client.embed_query(query)
    logger.debug(f"Generated embedding with {len(query_embedding)} dimensions")

    # Search using OpenSearch
    logger.debug("Searching OpenSearch")
    search_results = opensearch_client.search_unified(
        query=query,
        query_embedding=query_embedding,
        size=top_k,
        use_hybrid=use_hybrid,
    )

    # Convert SearchHit to LangChain Document
    documents = []
    hits = search_results.get("hits", [])
    logger.info(f"Found {len(hits)} documents from OpenSearch")

    for hit in hits:
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


if __name__ == "__main__":
    opensearch_client = make_opensearch_client()
    print('Health check:', opensearch_client.health_check())
    mcp.run(
        transport="streamable-http",   # hoặc "http" nếu version mới dùng tên này
        host="0.0.0.0",                # Quan trọng: bind tất cả interface (để Docker truy cập)
        port=8100,                     # Phải khớp với EXPOSE trong Dockerfile
        path="/mcp"                    # Đảm bảo endpoint là /mcp
    )

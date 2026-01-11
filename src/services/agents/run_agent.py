# run_agent.py

import asyncio

from src.services.langfuse.factory import make_langfuse_tracer
from src.services.embeddings.factory import make_embeddings_client
from src.services.nvidia.factory import make_nvidia_client
from src.services.opensearch.factory import make_opensearch_client
from src.services.agents.factory import make_agentic_rag_service

from src.services.nvidia.client import NvidiaClient
from src.services.embeddings.jina_client import JinaEmbeddingsClient

async def main():

    opensearch_client = make_opensearch_client()  
    nvidia_client = make_nvidia_client()
    embeddings_client = make_embeddings_client()
    langfuse=make_langfuse_tracer()
    service = make_agentic_rag_service(
        opensearch_client=opensearch_client,
        nvidia_client=nvidia_client,
        embeddings_client=embeddings_client,
        langfuse_tracer=langfuse,
        top_k=6,
        use_hybrid=True,
    )
    for i in range(2):
        query = input("\nCâu hỏi về paper arXiv: ").strip()
        result = await service.ask(query=query)


if __name__ == "__main__":
    asyncio.run(main())

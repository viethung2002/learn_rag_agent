import logging
import time
from typing import Dict,List, Any

from langgraph.runtime import Runtime
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage

from ..context import Context
from ..models import GradeDocuments, GradingResult
from ..prompts import GRADE_DOCUMENTS_PROMPT
from ..state import AgentState
from .utils import get_latest_context, get_latest_query

logger = logging.getLogger(__name__)


async def ainvoke_rerank_documents_step(
        state: AgentState, 
        runtime:  Runtime[Context]
) -> Dict[str, str]:
    """
    Node rerank documents bằng NVIDIA reranker.
    Input: List[Document] từ retrieved_docs
    Output: List[Document] đã rerank, có thêm relevance_score trong metadata
    """
    logger.info("NODE: rerank")

    documents_dict = state["retrieved_docs"]
    documents=[Document(page_content=doc['page_content'], metadata=doc['metadata']) for doc in documents_dict]
    logger.info(f"Number of documents: {len(documents)}")
    if not documents:
        logger.warning("Không có documents để rerank")
        return {
            "reranked_docs": [],
            "routing_decision": "grade_documents" 
        }

    question = get_latest_query(state["messages"])

    try:
        # Lấy reranker từ NvidiaClient
        reranker = runtime.context.nvidia_client.get_reranker(
            model="nvidia/llama-3.2-nv-rerankqa-1b-v2",
            top_n=8  # có thể lấy từ graph_config nếu cần
        )

        logger.warning(f"Reranking {len(documents)} documents...")
        logger.warning(f"Reranker class: {reranker.__class__.__name__}")
        # Rerank → trả về List[Document] đã sắp xếp, có score trong metadata
        reranked_docs = await reranker.acompress_documents(
            documents=documents,
            query=question
        )
        logger.warning("metadata của document sau rerank:")
        for i, doc in enumerate(reranked_docs):
            logger.info(f"Document {i}: {doc.metadata}")
        # (Tùy chọn) Thêm log điểm số
        scores = [doc.metadata.get("relevance_score", 0.0) for doc in reranked_docs]
        if scores:
            logger.info(f"Max relevance score sau rerank: {max(scores):.3f}")

        # Quyết định routing cơ bản (có thể tinh chỉnh sau)
        # has_good_docs = any(s >= 1.0 for s in scores) if scores else False
        # routing = "generate_answer" if has_good_docs else "rewrite_query"
        
        filter_top2_docs = reranked_docs[:2]
        state["messages"].append(ToolMessage(content=filter_top2_docs, tool_call_id='Rerank'))
    
        return {
            "reranked_docs": filter_top2_docs,          # List[Document] đã rerank
            "retrieved_docs": filter_top2_docs,         # cập nhật lại field này cho node sau dùng
            "rerank_scores": scores,
            # "routing_decision": routing
        }

    except Exception as e:
        logger.error(f"Rerank thất bại: {str(e)}", exc_info=True)
        return {
            "reranked_docs": documents,
            "retrieved_docs": documents,
            "rerank_scores": [0.0] * len(documents),
        }

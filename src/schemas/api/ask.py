from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request model for RAG question answering."""

    query: str = Field(..., description="User's question", min_length=1, max_length=1000)
    top_k: int = Field(3, description="Number of top chunks to retrieve", ge=1, le=10)
    use_hybrid: bool = Field(True, description="Use hybrid search (BM25 + vector)")
    model: str = Field("llama3.2:1b", description="Nvidia model to use for generation")
    categories: Optional[List[str]] = Field(None, description="Filter by arXiv categories")
    thread_id: Optional[str] = Field(None, description="Thread ID for memory management")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are transformers in machine learning?",
                "top_k": 3,
                "use_hybrid": True,
                "model": "openai/gpt-oss-120b",
                "categories": ["cs.AI", "cs.LG"],
                "thread_id": "1234567890",
            }
        }


class AskResponse(BaseModel):
    """Response model for RAG question answering."""

    query: str = Field(..., description="Original user question")
    answer: str = Field(..., description="Generated answer from LLM")
    sources: List[str] = Field(..., description="PDF URLs of source papers")
    chunks_used: int = Field(..., description="Number of chunks used for generation")
    search_mode: str = Field(..., description="Search mode used: bm25 or hybrid")
    evaluation: Optional["EvaluationSummary"] = Field(None, description="Optional automated evaluation scores")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are transformers in machine learning?",
                "answer": "Transformers are a neural network architecture...",
                "sources": ["https://arxiv.org/pdf/1706.03762.pdf", "https://arxiv.org/pdf/1810.04805.pdf"],
                "chunks_used": 3,
                "search_mode": "hybrid",
            }
        }


class AgenticAskResponse(AskResponse):
    """Response model for agentic RAG question answering."""

    reasoning_steps: List[str] = Field(..., description="Agent's decision-making steps")
    retrieval_attempts: int = Field(..., description="Number of document retrieval attempts")
    trace_id: Optional[str] = Field(None, description="Langfuse trace ID for feedback and debugging")
    thread_id: Optional[str] = Field(
        None,
        description="Conversation thread id (persisted history); send on the next turn to continue",
    )
    rewritten_query: Optional[str] = Field(
        None,
        description="If the agent rewrote the query before retrieval",
    )
    neo4j_attempted: bool = Field(
        False,
        description="Whether the agent actually queried Neo4j for graph facts during this request",
    )
    used_neo4j: bool = Field(
        False,
        description="Whether Neo4j graph facts were attached to at least one retrieved document",
    )
    graph_enriched_docs: int = Field(
        0,
        description="Number of retrieved documents enriched with Neo4j graph facts",
    )
    graph_enriched_arxiv_ids: List[str] = Field(
        default_factory=list,
        description="arXiv IDs of retrieved documents that were enriched from Neo4j",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are transformers in machine learning?",
                "answer": "Transformers are neural network architectures...",
                "sources": ["https://arxiv.org/pdf/1706.03762.pdf"],
                "chunks_used": 3,
                "search_mode": "hybrid",
                "reasoning_steps": [
                    "Decided to retrieve relevant papers",
                    "Retrieved documents from database",
                    "Generated answer from relevant documents",
                ],
                "retrieval_attempts": 1,
                "trace_id": "abc123-def456-ghi789",
                "thread_id": "550e8400-e29b-41d4-a716-446655440000",
                "rewritten_query": None,
                "neo4j_attempted": True,
                "used_neo4j": True,
                "graph_enriched_docs": 1,
                "graph_enriched_arxiv_ids": ["1706.03762"],
            }
        }


class FeedbackRequest(BaseModel):
    """Request model for user feedback on RAG answers."""

    trace_id: str = Field(..., description="Langfuse trace ID from the response")
    score: float = Field(..., description="Feedback score (0-1 or -1 to 1)", ge=-1, le=1)
    comment: Optional[str] = Field(None, description="Optional feedback comment", max_length=1000)

    class Config:
        json_schema_extra = {
            "example": {
                "trace_id": "abc123-def456-ghi789",
                "score": 1.0,
                "comment": "This answer was very helpful and accurate!",
            }
        }


class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""

    success: bool = Field(..., description="Whether feedback was recorded successfully")
    message: str = Field(..., description="Status message")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Feedback recorded successfully",
            }
        }


class LLMJudgeResult(BaseModel):
    score: float = Field(..., description="Judge score normalized to 0-1", ge=0, le=1)
    verdict: str = Field(..., description="Short verdict from the judge")
    reasoning: Optional[str] = Field(None, description="Judge explanation")


class EvaluationSummary(BaseModel):
    ragas_metrics: Dict[str, float] = Field(default_factory=dict, description="RAGAS metric scores by name")
    llm_judge: Optional[LLMJudgeResult] = Field(None, description="LLM-as-judge result")
    reference_used: bool = Field(False, description="Whether a reference answer was used")
    status: str = Field("completed", description="Evaluation execution status")


AskResponse.model_rebuild()
AgenticAskResponse.model_rebuild()


class WebSearchConsentRequest(BaseModel):
    """Request to run web search after user consent."""

    thread_id: Optional[str] = Field(None, description="Conversation thread id")
    query: str = Field(..., description="Original user query", min_length=1, max_length=1000)
    consent: bool = Field(..., description="User consent to run web search")


class WebSearchConsentResponse(BaseModel):
    success: bool = Field(..., description="Whether the web search was executed")
    summary: Optional[str] = Field(None, description="Web search summary if executed")
    message: Optional[str] = Field(None, description="Info message")


WebSearchConsentResponse.model_rebuild()

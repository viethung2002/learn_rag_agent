import logging
import uuid

from fastapi import APIRouter, HTTPException, Query
from src.dependencies import AgenticRAGDep, CacheDep, EvaluationDep, LangfuseDep
from src.schemas.api.ask import AgenticAskResponse, AskRequest, FeedbackRequest, FeedbackResponse
from src.schemas.api.ask import WebSearchConsentRequest, WebSearchConsentResponse
from src.schemas.api.agent_chat import (
    AgentChatConversationPublic,
    AgentChatConversationsResponse,
    AgentChatMessagePublic,
    AgentChatMessagesResponse,
)
from src.api.deps import CurrentUser, SessionDep
from src.crud import agent_chat as agent_chat_crud
from langchain_mcp_adapters.tools import load_mcp_tools

router = APIRouter(prefix="/api/v1", tags=["agentic-rag"])
logger = logging.getLogger(__name__)


@router.post("/ask-agentic", response_model=AgenticAskResponse)
async def ask_agentic(
    request: AskRequest,
    agentic_rag: AgenticRAGDep,
    cache_client: CacheDep,
    evaluation_service: EvaluationDep,
    session: SessionDep, current_user: CurrentUser, skip: int = 0, limit: int = 100
) -> AgenticAskResponse:
    """
    Agentic RAG endpoint with intelligent retrieval and query refinement.

    Features:
    - Decides if retrieval is needed
    - Grades document relevance
    - Rewrites queries if needed
    - Provides reasoning transparency

    The agent will automatically:
    1. Determine if the question requires research paper retrieval
    2. If needed, search for relevant papers
    3. Grade retrieved documents for relevance
    4. Rewrite the query if documents aren't relevant
    5. Generate an answer with citations

    Args:
        request: Question and parameters
        agentic_rag: Injected agentic RAG service

    Returns:
        Answer with sources and reasoning steps

    Raises:
        HTTPException: If processing fails
    """
    try:
        thread_id = request.thread_id or str(uuid.uuid4())

        if cache_client:
            try:
                cached_response = await cache_client.find_cached_response(request)
                if cached_response:
                    logger.info("Returning cached response for exact query match")
                    data = dict(cached_response)
                    data["thread_id"] = thread_id
                    response = AgenticAskResponse.model_validate(data)
                    try:
                        agent_chat_crud.append_turn(
                            session,
                            user_id=current_user.id,
                            thread_id=thread_id,
                            user_content=request.query,
                            assistant_content=response.answer,
                            trace_id=response.trace_id,
                            extra={
                                "sources": response.sources,
                                "reasoning_steps": response.reasoning_steps,
                                "retrieval_attempts": response.retrieval_attempts,
                                "cache_hit": True,
                            },
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to persist agent chat history for cached response: %s",
                            e,
                        )
                        session.rollback()
                    return response
            except Exception as e:
                logger.warning(f"Cache check failed, proceeding with normal flow: {e}")

        result = await agentic_rag.ask(
            query=request.query,
            model=request.model,
            thread_id=thread_id,
        )

        evaluation = None
        if evaluation_service:
            evaluation = await evaluation_service.evaluate_answer(
                query=result["query"],
                answer=result["answer"],
                contexts=result.get("retrieved_contexts", []),
                metadata={
                    "endpoint": "ask-agentic",
                    "model": request.model,
                    "thread_id": thread_id,
                },
            )

        response = AgenticAskResponse(
            query=result["query"],
            answer=result["answer"],
            sources=result.get("sources", []),
            chunks_used=request.top_k,
            search_mode="hybrid" if request.use_hybrid else "bm25",
            reasoning_steps=result.get("reasoning_steps", []),
            retrieval_attempts=result.get("retrieval_attempts", 0),
            trace_id=result.get("trace_id"),
            thread_id=thread_id,
            rewritten_query=result.get("rewritten_query"),
            graph_retrieval_attempted=result.get("graph_retrieval_attempted", False),
            graph_retrieval_used=result.get("graph_retrieval_used", False),
            neo4j_enrichment_attempted=result.get("neo4j_enrichment_attempted", False),
            neo4j_enrichment_used=result.get("neo4j_enrichment_used", False),
            graph_enriched_docs=result.get("graph_enriched_docs", 0),
            graph_enriched_arxiv_ids=result.get("graph_enriched_arxiv_ids", []),
            evaluation=evaluation,
        )

        try:
            agent_chat_crud.append_turn(
                session,
                user_id=current_user.id,
                thread_id=thread_id,
                user_content=request.query,
                assistant_content=result["answer"],
                trace_id=result.get("trace_id"),
                extra={
                    "sources": result.get("sources"),
                    "reasoning_steps": result.get("reasoning_steps"),
                    "retrieval_attempts": result.get("retrieval_attempts"),
                    "graph_retrieval_attempted": result.get("graph_retrieval_attempted", False),
                    "graph_retrieval_used": result.get("graph_retrieval_used", False),
                    "neo4j_enrichment_attempted": result.get("neo4j_enrichment_attempted", False),
                    "neo4j_enrichment_used": result.get("neo4j_enrichment_used", False),
                    "graph_enriched_docs": result.get("graph_enriched_docs", 0),
                    "graph_enriched_arxiv_ids": result.get("graph_enriched_arxiv_ids", []),
                },
            )
        except Exception as e:
            logger.warning("Failed to persist agent chat history: %s", e)
            session.rollback()

        if cache_client:
            try:
                await cache_client.store_response(request, response)
            except Exception as e:
                logger.warning(f"Failed to store response in cache: {e}")

        return response

    except ValueError as e:  
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@router.get("/agentic/conversations", response_model=AgentChatConversationsResponse)
def list_agentic_conversations(
    session: SessionDep,
    current_user: CurrentUser,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
) -> AgentChatConversationsResponse:
    """List agent chat threads for the current user (newest first)."""
    total = agent_chat_crud.count_user_conversations(
        session, user_id=current_user.id
    )
    rows = agent_chat_crud.list_conversations(
        session, user_id=current_user.id, skip=skip, limit=limit
    )
    return AgentChatConversationsResponse(
        data=[AgentChatConversationPublic.model_validate(r) for r in rows],
        total=total,
    )


@router.get(
    "/agentic/conversations/{thread_id}/messages",
    response_model=AgentChatMessagesResponse,
)
def get_agentic_conversation_messages(
    thread_id: str,
    session: SessionDep,
    current_user: CurrentUser,
) -> AgentChatMessagesResponse:
    """Load persisted user/assistant messages for one thread."""
    tid = thread_id.strip()
    if not tid or len(tid) > 255:
        raise HTTPException(
            status_code=422,
            detail="thread_id must be 1–255 characters",
        )
    msgs = agent_chat_crud.list_messages_for_thread(
        session, user_id=current_user.id, thread_id=tid
    )
    if msgs is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return AgentChatMessagesResponse(
        thread_id=tid,
        messages=[AgentChatMessagePublic.model_validate(m) for m in msgs],
    )


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    langfuse_tracer: LangfuseDep,
    session: SessionDep, 
    current_user: CurrentUser, 
    skip: int = 0, 
    limit: int = 100
) -> FeedbackResponse:
    """
    Submit user feedback for an agentic RAG response.

    This endpoint allows users to rate the quality of answers and provide
    optional comments. Feedback is tracked in Langfuse for continuous improvement.

    Args:
        request: Feedback data including trace_id, score, and optional comment
        langfuse_tracer: Injected Langfuse tracer service

    Returns:
        FeedbackResponse indicating success or failure

    Raises:
        HTTPException: If feedback submission fails
    """
    try:
        if not langfuse_tracer:
            raise HTTPException(
                status_code=503,
                detail="Langfuse tracing is disabled. Cannot submit feedback."
            )

        success = langfuse_tracer.submit_feedback(
            trace_id=request.trace_id,
            score=request.score,
            comment=request.comment,
        )

        if success:
            # Flush to ensure feedback is sent immediately
            langfuse_tracer.flush()

            return FeedbackResponse(
                success=True,
                message="Feedback recorded successfully"
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to submit feedback to Langfuse"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error submitting feedback: {str(e)}"
        )


@router.post("/human-review/should-retrieve")
async def human_review_should_retrieve(
):
    """
    Human-in-the-loop review for should_retrieve decision
    """


@router.post("/agentic/web-search-consent", response_model=WebSearchConsentResponse)
async def agentic_web_search_consent(
    request: WebSearchConsentRequest,
    agentic_rag: AgenticRAGDep,
    session: SessionDep,
    current_user: CurrentUser,
):
    """Run the MCP `web_search` tool after explicit user consent.

    The frontend should call this endpoint when it receives `needs_web_search`.
    """
    try:
        if not request.consent:
            return WebSearchConsentResponse(success=False, summary=None, message="User declined web search")

        # Open MCP session and load tools
        async with agentic_rag.mcp_client.session("arxiv-tools") as mcp_session:
            tools = await load_mcp_tools(mcp_session)
            tools_by_name = {t.name: t for t in tools}

            web_tool = tools_by_name.get("web_search")
            if not web_tool:
                return WebSearchConsentResponse(success=False, summary=None, message="web_search tool not available")

            # Invoke the web_search tool
            web_res = await web_tool.ainvoke({"query": request.query})

            # Normalize summary
            summary = web_res if isinstance(web_res, str) else str(web_res)

            # Persist as assistant turn in chat history
            try:
                agent_chat_crud.append_turn(
                    session,
                    user_id=current_user.id,
                    thread_id=request.thread_id or "",
                    user_content=f"[User consented to web search] {request.query}",
                    assistant_content=summary,
                    trace_id=None,
                    extra={"source": "web_search"},
                )
            except Exception:
                session.rollback()

            return WebSearchConsentResponse(success=True, summary=summary, message="Web search completed")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run web search: {e}")

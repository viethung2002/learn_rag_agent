import logging
import uuid

from fastapi import APIRouter, HTTPException, Query
from src.dependencies import AgenticRAGDep, LangfuseDep, CacheDep
from src.schemas.api.ask import AgenticAskResponse, AskRequest, FeedbackRequest, FeedbackResponse
from src.schemas.api.agent_chat import (
    AgentChatConversationPublic,
    AgentChatConversationsResponse,
    AgentChatMessagePublic,
    AgentChatMessagesResponse,
    AgentChatThreadIdsResponse,
)
from src.api.deps import CurrentUser, SessionDep
from src.crud import agent_chat as agent_chat_crud

router = APIRouter(prefix="/api/v1", tags=["agentic-rag"])
logger = logging.getLogger(__name__)


@router.post("/ask-agentic", response_model=AgenticAskResponse)
async def ask_agentic(
    request: AskRequest,
    agentic_rag: AgenticRAGDep,
    cache_client: CacheDep,
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
    "/agentic/conversations/thread_ids",
    response_model=list[str],
)
def list_agentic_conversation_thread_ids(
    session: SessionDep,
    current_user: CurrentUser,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
) -> list[str]:
    """Return thread_id values for the current user (newest first)."""
    return agent_chat_crud.list_thread_ids(session, user_id=current_user.id, skip=skip, limit=limit)


@router.get("/agentic/thread-ids", response_model=AgentChatThreadIdsResponse)
def list_agentic_thread_ids(
    session: SessionDep,
    current_user: CurrentUser,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
) -> AgentChatThreadIdsResponse:
    """List persisted thread_ids for the current user."""
    rows = agent_chat_crud.list_conversations(
        session, user_id=current_user.id, skip=skip, limit=limit
    )
    total = agent_chat_crud.count_user_conversations(
        session, user_id=current_user.id
    )
    return AgentChatThreadIdsResponse(
        thread_ids=[row.thread_id for row in rows],
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

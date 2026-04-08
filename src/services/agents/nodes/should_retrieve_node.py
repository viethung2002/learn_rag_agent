import logging
import time
from typing import Dict, List, Any

from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from src.services.agents.models import ShouldRetrieveDecision
from src.services.agents.nodes.utils import get_latest_query, get_old_message

from ..context import Context
from ..prompts import REWRITE_PROMPT, SHOULD_RETRIEVE_PROMPT
from ..state import AgentState

logger = logging.getLogger(__name__)

_FORCE_RETRIEVE_MARKERS = (
    "citation",
    "citations",
    "reference",
    "references",
    "bibliography",
    "shared citations",
    "shared references",
    "common citations",
    "common references",
    "cited by",
)





def route_after_should_retrieve(state: AgentState) -> str:
    """Route based on should_retrieve decision"""
    decision = state.get("should_retrieve_result")
    if decision and not decision.should_retrieve:
        logger.info("LLM decided: No retrieval needed → direct to answer generation")
        return "generate_answer"
    logger.info("LLM decided: Retrieval needed → proceed to retrieve")
    return "retrieve"

async def ainvoke_should_retrieve_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, Any]:
    """Use LLM to decide whether to retrieve documents or answer directly."""
    
    query = get_latest_query(state["messages"])
    old_message = get_old_message(state["messages"])
    lowered = query.lower()

    if any(marker in lowered for marker in _FORCE_RETRIEVE_MARKERS):
        decision = ShouldRetrieveDecision(
            should_retrieve=True,
            reason="Forced retrieval for citation/reference graph query.",
        )
        logger.info("Should-retrieve heuristic forced retrieval: %s", decision.reason)
        return {"should_retrieve_result": decision}
    
    # Prompt yêu cầu LLM quyết định có cần retrieve hay không
    should_retrieve_prompt = SHOULD_RETRIEVE_PROMPT.format(question=query, old_message=old_message)
    
    llm = runtime.context.nvidia_client.get_langchain_model(
        model=runtime.context.model_name,
        temperature=0.0,
    )
    
    # Structured output: yes/no + reason
    structured_llm = llm.with_structured_output(ShouldRetrieveDecision)
    
    try:
        decision: ShouldRetrieveDecision = await structured_llm.ainvoke(should_retrieve_prompt)
        logger.info(f"Should-retrieve decision: {decision.should_retrieve}, Reason: {decision.reason}")
    except Exception as e:
        logger.warning(f"Should-retrieve LLM failed: {e}, defaulting to retrieve")
        decision = ShouldRetrieveDecision(should_retrieve=True, reason="Fallback due to LLM error")
    
    return {"should_retrieve_result": decision}

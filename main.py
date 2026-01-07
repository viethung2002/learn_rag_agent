api_key = ''

from typing import TypedDict, List

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
)
from langchain_core.messages.utils import (
    trim_messages,
    count_tokens_approximately,
)

from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import InMemorySaver

from langchain_nvidia_ai_endpoints import ChatNVIDIA


# =========================
# 1. Initialize LLM
# =========================
model = ChatNVIDIA(
    nvidia_api_key=api_key,
    model="nvidia/nemotron-3-nano-30b-a3b",
    temperature=0.7,
    top_p=0.9,
    max_completion_tokens=128,
)



# =========================
# 2. Define Manual State
# =========================
class ChatState(TypedDict):
    messages: List[BaseMessage]
    temp: int


# =========================
# 3. Graph Node (manual state handling)
# =========================
def call_model(state: ChatState) -> ChatState:
    # Trim history before sending to LLM
    trimmed_messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=128000,
        start_on="human",
    )

    # Call model
    response: AIMessage = model.invoke(trimmed_messages)

    # Add AI response
    state["messages"].append(response)
    state["temp"] = 1



# =========================
# 4. Build LangGraph
# =========================
checkpointer = InMemorySaver()

builder = StateGraph(ChatState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")

graph = builder.compile(checkpointer=checkpointer)


# =========================
# 5. Run Conversation (Manual)
# =========================
config = {
    "configurable": {
        "thread_id": "1"  # change this to reset memory
    }
}

# ---- Turn 1
state = {
    "messages": [
        HumanMessage(content="Hi, my name is Bob.")
    ]
}

result = graph.invoke(state, config)
result["messages"][-1].pretty_print()

# # ---- Turn 2
# state = {
#     "messages": result["messages"] + [
#         HumanMessage(content="Write a short poem about cats.")
#     ]
# }

# result = graph.invoke(state, config)
# result["messages"][-1].pretty_print()

# # ---- Turn 3
# state = {
#     "messages": result["messages"] + [
#         HumanMessage(content="Now do the same but for dogs.")
#     ]
# }

# result = graph.invoke(state, config)
# result["messages"][-1].pretty_print()

# ---- Turn 4 (memory test)
state = {
    "messages": result["messages"] + [
        HumanMessage(content="What's my name?")
    ]
}
result = graph.invoke(state, config)
result["messages"][-1].pretty_print()

print('final result', result)

msgs = list(graph.get_state_history(config))
print(msgs)

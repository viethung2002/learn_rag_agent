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
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import InMemorySaver

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from typing import TypedDict, Annotated, List
# =========================
# 1. Initialize LLM
# =========================
model = ChatNVIDIA(
    nvidia_api_key=api_key,
    model="nvidia/nemotron-3-nano-30b-a3b",
    temperature=0.7,
    top_p=0.9,
    max_completion_tokens=512,
)



# =========================
# 2. Define Manual State
# =========================
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    temp: int


# =========================
# 3. Graph Node (manual state handling)
# =========================
def call_model(state: ChatState) -> ChatState:
    # Trim history before sending to LLM
    print("---call model ----")
    print(f"current msg state {state['messages']}")
    print("------")
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
    return state



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
        HumanMessage(content="Hi, my name is Bob. I'm 18 years old."),
    ],
    "temp": 1
}

result = graph.invoke(state, config)
result["messages"][-1].pretty_print()
print('------------------')

input_2 = {"messages": [HumanMessage(content="What's my name?")]}
result_2 = graph.invoke(input_2, config)
result_2["messages"][-1].pretty_print()

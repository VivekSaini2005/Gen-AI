from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import TypedDict
from typing import Optional, Literal

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

llm1 = ChatGoogleGenerativeAI(
    model= "gemini-3-flash-preview",
    temperature=1.0,
    max_retries=2,
    google_api_key=api_key,
)

llm2 = ChatGoogleGenerativeAI(
    model= "gemini-2.5-flash",
    temperature=1.0,
    max_retries=2,
    google_api_key=api_key,
)
class State(TypedDict):
    user_query: str
    llm_output: Optional[str]
    is_good: Optional[bool]


def chatbot(state: State):
    response = llm1.invoke([
        {"role": "user","content": state.get("user_query")}
    ])

    state["llm_output"] = response.content
    print("\n\nExecuted gemini-3-flash-preview:", state)
    return state


def evaluate_response(state: State) -> Literal["endnode", "chatbot2"]:
    if False:
        return "endnode"
    return "chatbot2"


def chatbot2(state: State):
    response = llm2.invoke([
        {"role": "user","content": state.get("user_query")}
    ])

    state["llm_output"] = response.content
    print("\n\nExecuted gemini-2.5-flash:", state)
    return state

def endnode(state: State):
    print("\n\nExecuting endnode:", state)
    return state


graph_builder = StateGraph(State)


graph_builder.add_node("chatbot",chatbot)
graph_builder.add_node("chatbot2",chatbot2)
graph_builder.add_node("endnode",endnode)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", evaluate_response)

graph_builder.add_edge("chatbot2","endnode")
graph_builder.add_edge("endnode", END)

graph = graph_builder.compile()
updated_state = graph.invoke(State({"user_query":"What is 2 + 2?"}))
print("\n\nFinal State:",updated_state)

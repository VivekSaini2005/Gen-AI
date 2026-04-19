from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")


# Create LLM class
llm = ChatGoogleGenerativeAI(
    model= "gemini-3-flash-preview",
    temperature=1.0,
    max_retries=2,
    google_api_key=api_key,
)

class State(TypedDict):
    messages: Annotated[list,add_messages]


graph_builder = StateGraph(State)

def chatbot(state: State):
    response = llm.invoke(state.get("messages"))
    return {"messages": [response]}

def samplemodel(state: State):
    print("\n\nExecuting samplemodel node",state)
    return {"messages": ["Hi, This is message from sample model"]} 

graph_builder.add_node("chatbot",chatbot)    # add_node("name of the node", function that will be called when this node is executed)
graph_builder.add_node("samplemodel", samplemodel)

graph_builder.add_edge(START, "chatbot")   # add_edge("from node", "to node")
graph_builder.add_edge("chatbot", "samplemodel")
graph_builder.add_edge("samplemodel", END)

# (Start) --> (chatbot) --> (samplemodel) --> (End)

graph = graph_builder.compile()   # compile the graph to get the final graph object that can be executed

updated_state = graph.invoke(State({"messages":["Hi, My name is Vivek Saini and this is initial message"]}))   # invoke the graph with the initial state, it will execute the nodes in the order defined by the edges and return the final state after execution
print("\n\nFinal State:",updated_state)
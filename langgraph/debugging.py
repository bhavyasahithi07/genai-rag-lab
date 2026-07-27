from dotenv import load_dotenv

load_dotenv()

from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# -----------------------------
# LLM
# -----------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)


# -----------------------------
# Tools
# -----------------------------

tavily_tool = TavilySearch(max_results=2)


@tool
def multiply(a: int, b: int) -> str:
    """Multiply two numbers and return the result."""
    return f"Multiply result is {a * b}"


tools = [
    tavily_tool,
    multiply
]


llm_with_tools = llm.bind_tools(tools)


# -----------------------------
# Graph State
# -----------------------------

class State(TypedDict):
    messages: Annotated[list, add_messages]


# -----------------------------
# Nodes
# -----------------------------

def tool_calling_llm(state: State):
    response = llm_with_tools.invoke(
        state["messages"]
    )

    return {
        "messages": [response]
    }


# -----------------------------
# Build Graph
# -----------------------------

graph_builder = StateGraph(State)


graph_builder.add_node(
    "tool_calling_llm",
    tool_calling_llm
)

graph_builder.add_node(
    "tools",
    ToolNode(tools)
)


# -----------------------------
# Edges
# -----------------------------

graph_builder.add_edge(
    START,
    "tool_calling_llm"
)

graph_builder.add_conditional_edges(
    "tool_calling_llm",
    tools_condition
)

graph_builder.add_edge(
    "tools",
    "tool_calling_llm"
)


# IMPORTANT:
# No MemorySaver for LangGraph Dev/Cloud
graph = graph_builder.compile()
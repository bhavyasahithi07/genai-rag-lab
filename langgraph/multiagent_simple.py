from typing import TypedDict,Annotated,List,Literal
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage,SystemMessage
from langchain_core.tools import tool
from langchain_community.tools import TavilySearchResults
from langgraph.graph import StateGraph,START,END,MessagesState
from langgraph.prebuilt import create_react_agent,ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq

import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"]="true"

class AgentState(MessagesState):
    next_agent:str

@tool
def search_web(query:str)->str:
    """Search the web and give an answer"""
    search=TavilySearchResults(max_results=3)
    response1=search.invoke(query)
    return str(response1)

@tool
def write_summary(content:str)->str:
    """write summary of results gor from searching web"""
    return f"summary of result {content[:500]}"

llm=ChatGroq(model="llama-3.3-70b-versatile")

tools=[search_web,write_summary]

def reasearcher_agent(state:AgentState):
    """It searches the information in the internet"""
    messages=state["messages"]
    Sys_msg=SystemMessage(content="you are a helpful assistant who answers questions through internet search and returns it to summary writer tool")
    reasearcher_llm=llm.bind_tools([search_web])
    response2=reasearcher_llm.invoke([Sys_msg]+messages)
    return {
        "messages":[response2],
        "next_agent":"writer"
    }


def execute_tools(state:AgentState):
    """execute any pending tool calls"""
    messages=state["messages"]
    last_message=messages[-1]
    if hasattr(last_message,"tool_calls") and last_message.tool_calls:
        tool_node=ToolNode([search_web,write_summary])
        response3=tool_node.invoke(state)
        return response3
    return state


graph_builder=StateGraph(AgentState)

graph_builder.add_node("researcher",reasearcher_agent)
graph_builder.add_node("writer",execute_tools)

graph_builder.add_edge(START,"researcher")
graph_builder.add_edge("researcher","writer")
graph_builder.add_edge("writer",END)

final_workflow=graph_builder.compile()

final_response=final_workflow.invoke({"messages":"research about gen ai developers future. will it be replaced by ai?"})
print(final_response["messages"][-1].content)


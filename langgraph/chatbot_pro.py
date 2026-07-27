import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

from langchain_groq import ChatGroq
llm=ChatGroq(model="llama-3.3-70b-versatile")


from langchain_tavily import TavilySearch
tavily_tool=TavilySearch(max_results=2)

from langchain_core.tools import tool
@tool
def multiply(a:int,b:int)->int:
    """Multiply the numbers given and return the result"""
    return f"Multiply result is {a*b}"

tools=[tavily_tool,multiply]

llm_with_tool=llm.bind_tools(tools=tools)

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
class State(TypedDict):
    messages:Annotated[list,add_messages]


from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode,tools_condition
from langgraph.checkpoint.memory import MemorySaver
memory=MemorySaver()

def tool_calling_llm(state:State):
    return {"messages":[llm_with_tool.invoke(state["messages"])]}

graph_builder=StateGraph(State) #langgraph.graph.state.StateGraph
graph_builder.add_node("tool_calling_llm",tool_calling_llm)
graph_builder.add_node("tools",ToolNode(tools))

#edges
graph_builder.add_edge(START,"tool_calling_llm")
graph_builder.add_conditional_edges("tool_calling_llm",tools_condition)
graph_builder.add_edge("tools","tool_calling_llm")

#compile
graph=graph_builder.compile(checkpointer=memory)

config={"configurable":{"thread_id":"1"}}

result=graph.invoke({"messages":"my name is bhavya what is its meaning?"},config=config)
print(result["messages"][-1].content)

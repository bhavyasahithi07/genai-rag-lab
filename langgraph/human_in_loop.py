import os
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel,Field
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_TRACING"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

from langchain_groq import ChatGroq
llm=ChatGroq(model="llama-3.3-70b-versatile")

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
class State(TypedDict):
    messages:Annotated[list,add_messages]


from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode,tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langgraph.types import Command,interrupt
from langchain_core.messages import HumanMessage
memory=MemorySaver()

graph_builder=StateGraph(State)

@tool
def human_assistance(query:str)->str:
    """Request assistance from human"""
    human_response=interrupt({"query":query})
    return human_response["data"]
from langchain_tavily import TavilySearch
tavily_tool=TavilySearch(max_results=2)
tools=[tavily_tool,human_assistance]

llm_with_tools=llm.bind_tools(tools)

def chatbot(state:State):
    message=llm_with_tools.invoke(state["messages"])
    return {"messages":[message]}

graph_builder.add_node("chatbot",chatbot)
graph_builder.add_node("tools",ToolNode(tools))

graph_builder.add_conditional_edges("chatbot",tools_condition)
graph_builder.add_edge(START,"chatbot")
graph_builder.add_edge("tools","chatbot")

graph=graph_builder.compile(checkpointer=memory)

user_input="I need some expert guidance and assistance for building an AI agent.Could you request assitance to me?"
config={"configurable":{"thread_id":"test_1"}}

events=graph.stream(
    {"messages":[HumanMessage(content=user_input)]},
    config,stream_mode="values"
)

for event in events:
    if "messages" in event:
        event["messages"][-1]

human_response=("we are experts in assisting. we would recommend using langgraph to build your agent"
                "its more reliable than simple autonomous agents")
human_command=Command(resume={"data":human_response})
events=graph.stream(human_command,config,stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
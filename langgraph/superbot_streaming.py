from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages:Annotated[list,add_messages]

graph_builder=StateGraph(State) #langgraph.graph.state.StateGraph

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"]="true"

from langchain_ollama import ChatOllama
llm=ChatOllama(model="llama3")

from langgraph.checkpoint.memory import MemorySaver
memory=MemorySaver()

#Functionality node
def chatbot(state:State):
    return {"messages":[llm.invoke(state["messages"])]}

#add node
graph_builder.add_node("llm_chatbot",chatbot)

#add edge
graph_builder.add_edge(START,"llm_chatbot")
graph_builder.add_edge("llm_chatbot",END)

#compile
compilation=graph_builder.compile(checkpointer=memory)

config={"configurable":{"thread_id":"3"}}

for chunk in compilation.stream({"messages":"Hi my name is bhavya"},config=config,stream_mode="updates"):
    print(chunk)

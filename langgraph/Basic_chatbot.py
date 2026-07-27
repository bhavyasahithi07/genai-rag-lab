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

#Functionality node
def chatbot(state:State):
    return {"messages":[llm.invoke(state["messages"])]}

#add node
graph_builder.add_node("llm_chatbot",chatbot)

#add edge
graph_builder.add_edge(START,"llm_chatbot")
graph_builder.add_edge("llm_chatbot",END)

#compile
compilation=graph_builder.compile()

result=compilation.invoke({"messages":"Hi"})
#print(result)

#lets visualize graph
from IPython.display import Image,display
try:
    display(Image(compilation.get_graph().draw_mermaid_png()))
except Exception:
    pass







import os
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel,Field
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq
model=ChatGroq(model="llama-3.3-70b-versatile")

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langchain_core.tools import tool

@tool
def  search_hotels(city:str)->str:
    """search hotels and give a long list of hotels"""
    return f"""Hotels in {city}:
    1.Hilton
    2.Marriot
    3.Emirates"""

agent=create_agent(
    model=model,
    checkpointer=InMemorySaver(),
    tools=[search_hotels],
    middleware=[
        SummarizationMiddleware(
            model=model,
            trigger=("fraction",0.005),
            keep=("fraction",0.002)
        )
    ]
)

config={"configurable":{"thread_id":"test2"}}


def counting_tokens(messages):
    total_chars=0
    for m in messages:
        total_chars+=len(str(m.content))
    return total_chars//4
    
cities=["paris","new york","california","north carolina","dallas","fort worth","oklohoma"]
for city in cities:
    result=agent.invoke({"messages":[HumanMessage(content=f"Find hotels in {city}")]},config=config)
    tokens=counting_tokens(result["messages"])
    print(f"{city}:{tokens} tokens")
    print(result["messages"])

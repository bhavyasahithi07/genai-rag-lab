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


agent=create_agent(
    model=model,
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model=model,
            trigger=("messages",10),
            keep=("messages",4)
        )
    ]
)

config={"configurable":{"thread_id":"test1"}}

questions=[
    "what is 2+2",
    "what is 5+5",
    "what is 1+1",
    "what is 6+6",
    "what is 3+3",
    "what is 7+7",
    "what is 8+8",
    "what is 4+4"
    ]
for q in questions:
    result=agent.invoke({"messages":[HumanMessage(content=q)]},config=config)
    print(f"messages: {result}")
    print(f"messages:{len(result["messages"])}")
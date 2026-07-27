import os
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel,Field
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq
model=ChatGroq(model="llama-3.3-70b-versatile")

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool
@tool
def read_email_tool(email:str)->str:
    """A mock function to read an email by its id"""
    return f"content of email id {email}"
@tool
def send_email_tool(recipient:str,subject:str,body:str)->str:
    """A mock function to send an email to recipient"""
    return f"email is sent to {recipient} with subject {subject}"

agent=create_agent(
    model=model,
    tools=[read_email_tool,send_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email_tool":{"allowed_decisions":["approve","edit","reject"]},
                "read_email_tool":False
            }
        )
    ]

)

config={"configurable":{"thread_id":"test_1"}}

#step1 request
from langchain.messages import HumanMessage
response1=agent.invoke({
    "messages":[HumanMessage(content="send email to aytest@test.com with subject-hello and body-how are you")]
}, config=config)

#print(response1)

#step2 approve
from langgraph.types import Command
if "__interrupt__" in response1:
    print("Approving paused")
    response1=agent.invoke(Command(
    resume=[
        {
            "type": "approve"
        }
    ]
),config=config)
    print(response1["messages"][-1].content)
    #print(response1)



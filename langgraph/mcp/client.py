from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()

import asyncio
async def main():
    client=MultiServerMCPClient(
        {
            "Math":{
                "command":"python",
                "args":["mathserver.py"],
                "transport":"stdio"
            },
            "Weather":{
                "url":"http://localhost:8000/mcp",
                "transport":"streamable_http"
            }
        }
    )

    import os
    os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

    tools=await client.get_tools()
    model=ChatGroq(model="llama-3.3-70b-versatile")
    agent=create_agent(model=model,tools=tools)

    math_response=await agent.ainvoke(
        {
            "messages":[{"role":"user","content":"what is 2+2*5 and what is weather now in dallas?"}]
        }
    )

    print("Math response-",math_response["messages"][-1].content)

asyncio.run(main())
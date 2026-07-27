import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")


from langchain.agents import create_agent
from langchain_groq import ChatGroq
model=ChatGroq(model="llama-3.3-70b-versatile")

from langchain.tools import tool

@tool
def get_weather(city:str)->str:
    """Get weather update at the location"""
    return f"{city} is sunny today"

model_with_tools=model.bind_tools([get_weather])

response=model_with_tools.invoke("whats the weather in dallas?")
print(response.tool_calls)

#just a print in list
for tool_call in response.tool_calls:
    print(f"tool:{tool_call["name"]}")

#custom workflow
#1.model generate tool calls
messages=[{"role":"user","content":"whats the weather in fort woth"}]
ai_msg=model_with_tools.invoke(messages)
messages.append(ai_msg)

#2.execute tools and get results
for call in ai_msg.tool_calls:
    results=get_weather.invoke(call)
    messages.append(results)

#print results
final_result=model_with_tools.invoke(messages)
print(final_result.content)

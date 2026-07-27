import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")


from langchain.agents import create_agent
from langchain_groq import ChatGroq
model=ChatGroq(model="llama-3.3-70b-versatile")

def get_weather(city:str)->str:
    """Get weather of a city"""
    return f"{city} is sunny"

agent=create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="you are a helpful assistant"
)
response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What is the weather in New York?"
        }
    ]
})

print(response["messages"][-1].content)

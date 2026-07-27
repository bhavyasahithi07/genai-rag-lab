import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
#langsmith
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"]="true"

from langchain_groq import ChatGroq
llm=ChatGroq(model="openai/gpt-oss-20b",groq_api_key=os.getenv("GROQ_API_KEY"))

#from langchain_core.prompts import ChatPromptTemplate
#prompt=ChatPromptTemplate.from_messages([
 #   ("system","You are a teacher. Answer based on questions asked."),
  #  ("user","{input}")
#])
from langchain_core.messages import HumanMessage,SystemMessage
messages=[
    SystemMessage(content="Translate the human message from english to French"),
    HumanMessage(content="Hello how are you?")
]


from langchain_core.output_parsers import StrOutputParser
op=StrOutputParser()

chain=llm|op

response=chain.invoke(messages)
print(response)

from fastapi import FastAPI
from langserve import add_routes
app=FastAPI(title="langchain test", version="1.0",description="a simple test")

add_routes(app,chain,path="/chain")

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="127.0.0.1",port=8000)


import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GEMINI_API_KEY"]=os.getenv("GEMINI_API_KEY")
#langsmith tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

from langchain_google_genai import ChatGoogleGenerativeAI

llm=ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

#result=llm.invoke("what is the meaning of name- Bhavya according to hindu dharma?")
#print(result)

from langchain_core.prompts import ChatPromptTemplate
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","you are an AI Engineer. Provide me answer based on questions"),
        ("user","{input}")
    ]
)


#chain=prompt|llm
#response=chain.invoke({"input":"can you tell me about langsmith?"})

from langchain_core.output_parsers import StrOutputParser
output_parser=StrOutputParser()
chain=prompt|llm|output_parser
response=chain.invoke({"input":"can you tell me about langsmith?"})


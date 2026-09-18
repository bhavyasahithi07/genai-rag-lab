from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper

api_wrapper_arxiv=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250)
arxiv=ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

api_wrapper_wiki=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
loader=WebBaseLoader("https://docs.smith.langchain.com/")
docs=loader.load()
documents=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(docs)
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-V2")
vectordb=FAISS.from_documents(documents,embeddings)
retriever=vectordb.as_retriever()

from langchain_core.tools.retriever import create_retriever_tool
retriever_tool=create_retriever_tool(retriever,"langsmith_search","Searching any information about langsmith")

tools=[arxiv,wiki,retriever_tool]

from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
import os

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
llm=ChatGroq(model="llama-3.3-70b-versatile")

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

from langchain.agents import create_agent
agent=create_agent(model=llm,tools=tools,system_prompt="you are a helpful assistant")

"""
from langchain.agents import AgentExecutor
agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True) #for create_openai_tools
"""

response=agent.invoke({
    "messages":[(
        "user","what is langsmith?"
    )]
})


print(response)


for message in response["messages"]:
    if hasattr(message, "tool_calls") and message.tool_calls:
        for tool_call in message.tool_calls:
            print("Tool:", tool_call["name"])
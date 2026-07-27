import os
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGSMITH_TRACING"]="true"
os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")

from langchain_community.document_loaders import PyPDFLoader
loader=PyPDFLoader(r"C:\GenAi\practice\ragpdf.pdf")
docs=loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
splits=text_splitter.split_documents(docs)

from langchain_huggingface import HuggingFaceEmbeddings
embeddings=HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")

from langchain_chroma import Chroma
vector_store=Chroma.from_documents(documents=splits,embedding=embeddings)

retriever=vector_store.as_retriever()

def format_docs(documents):
   return "\n\n".join(doc.page_content for doc in documents)

from langchain_ollama import ChatOllama
llm=ChatOllama(model="llama3")

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage,SystemMessage
prompt=ChatPromptTemplate.from_messages([
    ("system","""Be a helpful assistant where you need to answer user questions based on pdf data only
     Context:{context}
     if answer is not in the context say "I dont know."""),
    ("human","{question}")
])

from langchain_core.runnables import RunnablePassthrough
chain=({"context":retriever | format_docs,
        "question":RunnablePassthrough()}
|prompt|llm)

response=chain.invoke("what is rag?")
print(response)

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGSMITH_TRACING"]="true"
os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")

from langchain_community.document_loaders import TextLoader
loader=TextLoader("C:\GenAi\LangChain\Data_Ingestion\speech.txt")
result=loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter=RecursiveCharacterTextSplitter(chunk_size=10,chunk_overlap=5)
splits=text_splitter.split_documents(result)

from langchain_huggingface import HuggingFaceEmbeddings
embeddings=HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")

from langchain_chroma import Chroma
vector_store=Chroma.from_documents(documents=splits,embedding=embeddings)
retriever=vector_store.as_retriever()


response=retriever.invoke("what is her name?")
print(response[-1].page_content)






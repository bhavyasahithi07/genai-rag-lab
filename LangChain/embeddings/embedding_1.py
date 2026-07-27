from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma


load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
) #here we can specify dimension too like dimensions=

loader=TextLoader('C:\GenAi\LangChain\Data_Ingestion\speech.txt')
text_documents=loader.load()

text_splitter=RecursiveCharacterTextSplitter(chunk_size=30,chunk_overlap=10)
output=text_splitter.split_documents(text_documents)

#text = "This is Bhavya Sahithi Indukuri starting Gen AI learning and wanted to get a job by Jan 2028"
#query_result = embeddings.embed_query(output)
#embeddings.embed_documents([doc.page_content for doc in output])

db=Chroma.from_documents(documents=output,embedding=embeddings)
print(db)

query="GENAI RAG Developer"
results=db.similarity_search(query)
print(results)
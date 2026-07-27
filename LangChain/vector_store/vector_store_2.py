from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
from dotenv import load_dotenv
load_dotenv()
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")

loader=TextLoader("C:\GenAi\LangChain\Data_Ingestion\speech.txt")
text=loader.load()

text_splitter=RecursiveCharacterTextSplitter(chunk_size=20,chunk_overlap=5)
docs=text_splitter.split_documents(text)

embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-V2")

db=Chroma.from_documents(documents=docs,embedding=embeddings)
#print(db)

#similarity search
query="what is her name?"
search_doc=db.similarity_search(query)
#print(search_doc[0].page_content)

#save db
db=Chroma.from_documents(documents=docs,embedding=embeddings,persist_directory="./vector_store")

#load db
db2=Chroma(persist_directory="./vector_store",embedding_function=embeddings)
final_docs=db2.similarity_search(query)
#print(final_docs)

retriever=db.as_retriever()
final_retrieval=retriever.invoke(query)
print(final_retrieval[0].page_content)
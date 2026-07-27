from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

import os
from dotenv import load_dotenv
load_dotenv()
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")

loader=TextLoader("C:\GenAi\LangChain\Data_Ingestion\speech.txt")
text=loader.load()

text_splitter=CharacterTextSplitter(chunk_size=10,chunk_overlap=5)
docs=text_splitter.split_documents(text)

embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-V2")

db=FAISS.from_documents(documents=docs,embedding=embeddings)
print(db)

#similarity search
query="What is Bhavya wanted to become?"
result=db.similarity_search(query)
#print(result[0].page_content)

#retriever
retriever=db.as_retriever()
final_retrieval=retriever.invoke(query)
print(final_retrieval[0].page_content)

#similarity search with score
result_and_score=db.similarity_search_with_score(query)
#print(result_and_score)

#similarity search by vector
embedding_vector=embeddings.embed_query(query)
vector_result=db.similarity_search_by_vector(embedding_vector)
print(vector_result)

#saving db
db.save_local("faiss_index")

#loading db
new_db=FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
result_from_new_db=db.similarity_search(query)
print(result_from_new_db)
from langchain_core.documents import Document

documents = [
    Document(
        page_content="Python is a popular programming language.",
        metadata={"source": "python_book", "page": 1}
    ),
    Document(
        page_content="LangChain helps build LLM applications.",
        metadata={"source": "langchain_docs", "page": 2}
    ),
    Document(
        page_content="FAISS is used for efficient vector similarity search.",
        metadata={"source": "faiss_docs", "page": 5}
    ),
    Document(
        page_content="Retrievers fetch relevant documents based on a query.",
        metadata={"source": "rag_notes", "chapter": 3}
    ),
    Document(
        page_content="Embeddings convert text into numerical vectors.",
        metadata={"source": "embedding_guide", "page": 10}
    ),
    Document(
        page_content="Ollama allows you to run open-source LLMs locally.",
        metadata={"source": "ollama_docs", "version": "latest"}
    ),
    Document(
        page_content="Groq provides fast inference for supported language models.",
        metadata={"source": "groq_docs", "section": "API"}
    ),
    Document(
        page_content="Prompt engineering improves the quality of LLM responses.",
        metadata={"source": "ai_course", "lesson": 4}
    ),
    Document(
        page_content="Vector databases store and retrieve embeddings efficiently.",
        metadata={"source": "vector_db_notes", "chapter": 6}
    ),
    Document(
        page_content="RAG combines retrieval with generation for accurate answers.",
        metadata={"source": "rag_book", "page": 25}
    )
]

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"]="true"

#step1- create llm
from langchain_ollama import ChatOllama
llm=ChatOllama(model="llama3")

from langchain_huggingface import HuggingFaceEmbeddings
embeddings=HuggingFaceEmbeddings(model="all-MiniLM-L6-V2")

from langchain_chroma import Chroma
vector_store=Chroma.from_documents(documents=documents,embedding=embeddings)
#print(vector_store)

response=vector_store.similarity_search("python")
#print(f"without score={response}")

#async query
async def main():
    response2=await vector_store.asimilarity_search("python")
    #print(f"async={response2}")

response_with_score=vector_store.similarity_search_with_score("python")
#print(f"with score= {response_with_score}")


from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

retriever=RunnableLambda(vector_store.similarity_search).bind(k=1)
result=retriever.batch(["python"])
#print(result[0][0].page_content)


#best method
retriever2=vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k":1}
)
result2=retriever2.batch(["python","embeddings"])
#print(result2)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
message="""
Answer the question based on the context only
{question}

context={context}

"""
prompt=ChatPromptTemplate.from_messages([("human",message)])
chain={"context":retriever2,"question":RunnablePassthrough()}|prompt|llm
final_response=retriever2.invoke("tell about python")
print(final_response)


import os
from dotenv import load_dotenv
from pinecone import Pinecone,ServerlessSpec
load_dotenv()

os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGSMITH_TRACING"]="true"
os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")

pc=Pinecone()

index_name = "rag-index"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)

from langchain_community.document_loaders import PyPDFLoader
loader=PyPDFLoader(r"C:\GenAi\practice\ragpdf.pdf")
docs=loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
splits=text_splitter.split_documents(docs)

from rank_bm25 import BM25Okapi
texts=[doc.page_content for doc in splits]
tokenized_texts=[text.lower().split() for text in texts]
bm25=BM25Okapi(tokenized_texts)


from langchain_huggingface import HuggingFaceEmbeddings
embeddings=HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")

from sentence_transformers import CrossEncoder
reranker=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


from langchain_pinecone import PineconeVectorStore
vector_store=PineconeVectorStore.from_documents(documents=splits,embedding=embeddings,index_name=index_name)

retriever=vector_store.as_retriever(search_kwargs={"k":5})
def hybrid_search(query):
   dense_docs=retriever.invoke(query)

   sparse_docs=bm25.get_top_n(
      query.lower().split(),
      splits,
      n=5
   )
   return dense_docs+sparse_docs

def rerank_documents(query,documents,top_k=3):
   pairs=[(query,document.page_content) for document in documents]
   scores=reranker.predict(pairs)
   ranked_documents=sorted(zip(documents,scores),key=lambda x:x[1],reverse=True)
   return [doc for doc,score in ranked_documents[:top_k]]

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

from langchain_core.runnables import RunnablePassthrough,RunnableLambda
chain=({"context":RunnableLambda(
    lambda q:rerank_documents(q,hybrid_search(q)))
        | RunnableLambda(format_docs),
         "question":RunnablePassthrough()}
 |prompt|llm)

response=chain.invoke("what is rag?")
print(response)

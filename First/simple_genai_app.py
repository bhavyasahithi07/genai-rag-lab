import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GEMINI_API_KEY"]=os.getenv("GEMINI_API_KEY")
#langsmith tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

from langchain_community.document_loaders import WebBaseLoader
loader=WebBaseLoader("https://docs.langchain.com/oss/python/langchain/overview")
documents=loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
docs=text_splitter.split_documents(documents)

from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings=GoogleGenerativeAIEmbeddings( model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY"))

from langchain_community.vectorstores import FAISS
vectorstore_db=FAISS.from_documents(docs,embeddings)
print(vectorstore_db)

query="what are the core benefits?"
result=vectorstore_db.similarity_search(query)
print(result[0].page_content)

from langchain_core.prompts import ChatPromptTemplate
prompt=ChatPromptTemplate.from_template(
    """
Answer the following question based only on provided context:
<context>
{context}
</context>

"""
)

from langchain_google_genai import ChatGoogleGenerativeAI
llm=ChatGoogleGenerativeAI(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

chain= prompt|llm


from langchain_core.documents import Document
response=chain.invoke({
    "input":"what are four core benefits of langchain",
    "context":[Document(page_content="There are four core benefits of langchain. They are- standard model interface, highly configurable harness, built on top of langchain, debug with langchain")]
})

print(response)

retriever=vectorstore_db.as_retriever()
inputs="what are four core benefits of langchain?"
result=retriever.invoke(inputs['answer'])

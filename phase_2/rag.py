import os
from dotenv import load_dotenv

load_dotenv()
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")


import bs4
from langchain_community.document_loaders import WebBaseLoader
loader=WebBaseLoader(
    web_path=("https://en.wikipedia.org/wiki/LangChain"),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            id=("mwQg")
        )
    )
)
docs=loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitters=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=20)
splits=text_splitters.split_documents(docs)

from langchain_huggingface import HuggingFaceEmbeddings
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

from langchain_chroma import Chroma
vector_store=Chroma.from_documents(documents=splits,embedding=embeddings)
retriever=vector_store.as_retriever()
print(retriever)

from langchain_ollama import ChatOllama
llm=ChatOllama(model="llama3")


from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
system_prompt=(
    "you are an assistant for question-answer tasks"
    "use the following pieces of retrieved context to answer"
    "the question.If you dont know the answer , say thank you"
    "dont know. Use three sentences maximum and keep the"
    "answer concise"
    "\n\n"
    "{context}"
)

prompt=ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ("human","{input}")
    ]
)

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
question_answer_chain=create_stuff_documents_chain(llm,prompt)
rag_chain=create_retrieval_chain(retriever,question_answer_chain)
response=rag_chain.invoke({"input":"what is the history of it?"})
#print(response)

from langchain_classic.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

context_system_prompt=(
    "given a chat history and latest user question"
    "which refers context in chat history"
    "formulate a standalone question which can be understood"
    "without chat history do not answer the question"
    "just reformulate it if needed otherwise return as is"
)

context_prompt=ChatPromptTemplate.from_messages(
    [
    ("system",context_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human","{input}"),
    ]
)

qa_prompt=ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human","{input}")
    ]
)

history_aware_retriever=create_history_aware_retriever(llm,retriever,context_prompt)
#print(history_aware_retriever)
question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)


from langchain_core.messages import AIMessage,HumanMessage
chat_history=[]
question="what is................................"
response1=rag_chain.invoke({"input":question,"chat_history":chat_history})

chat_history.extend(
    [
        HumanMessage(content=question),
        AIMessage(content=response1["answer"])
    ]
)

question2="tell me more"
response2=rag_chain.invoke({"chat_history":chat_history,"input":question2})
#print(response2)
print(chat_history)




# from langchain_core.output_parsers import StrOutputParser
# parser=StrOutputParser()

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# #3. Create RAG chain
# from langchain_core.runnables import RunnablePassthrough
# rag_chain = (
#     {
#         "context": retriever | format_docs,
#         "input": RunnablePassthrough()
#     }
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# # 4. Ask question
# response = rag_chain.invoke(
#     "What is LangChain?"
# )

# print(response)

# from langchain_classic.chains import create_history_aware_retriever





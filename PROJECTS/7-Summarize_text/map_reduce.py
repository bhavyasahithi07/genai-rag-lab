import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate

from langchain_groq import ChatGroq
api_key=os.getenv("GROQ_API_KEY")
llm=ChatGroq(groq_api_key=api_key,model="openai/gpt-oss-20b")

loader=PyPDFLoader("Genai.pdf")
pages=loader.load_and_split()

map_prompt = ChatPromptTemplate.from_template(
    "Summarize the following text concisely:\n\n{text}"
)

map_chain = map_prompt | llm

reduce_prompt = ChatPromptTemplate.from_template(
    "Create one final concise summary from these summaries:\n\n{text}"
)

reduce_chain = reduce_prompt | llm

summaries = []

for doc in pages:
    response = map_chain.invoke({
        "text": doc.page_content
    })
    summaries.append(response.content)

combined_summaries = "\n\n".join(summaries)

final_response = reduce_chain.invoke({
    "text": combined_summaries
})

print(final_response)
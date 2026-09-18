from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
api_key=os.getenv("GROQ_API_KEY")
llm=ChatGroq(groq_api_key=api_key,model="openai/gpt-oss-20b")

loader=PyPDFLoader("Genai.pdf")
pages=loader.load_and_split()

template="""
write a concise summary about the following speech
{context}
"""
prompt=PromptTemplate.from_template(template=template)

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
chain=create_stuff_documents_chain(llm=llm,prompt=prompt)
output=chain.invoke({"context":pages})
print(output)
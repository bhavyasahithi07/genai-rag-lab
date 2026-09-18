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

from langchain_core.prompts import ChatPromptTemplate

# First chunk
initial_prompt = ChatPromptTemplate.from_template(
    """Write a concise summary of the following text:

{text}"""
)

initial_chain = initial_prompt | llm


# Subsequent chunks
refine_prompt = ChatPromptTemplate.from_template(
    """We have an existing summary:

{existing_summary}

Here is additional text:

{text}

Refine the existing summary using the new information.
Keep it concise."""
)

refine_chain = refine_prompt | llm

# First chunk
summary = initial_chain.invoke({
    "text": pages[0].page_content
}).content

# Remaining chunks
for page in pages[1:]:
    summary = refine_chain.invoke({
        "existing_summary": summary,
        "text": page.page_content
    }).content


print(summary)
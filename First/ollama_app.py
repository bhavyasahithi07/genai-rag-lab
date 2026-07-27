import os
from dotenv import load_dotenv

from langchain_ollama import OllamaLLM
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","you are a helpful love advisor. Provide me answers based on questions"),
        ("user","Question:{question}")
    ]
)


st.title("Langchain demo with Llama3 model")
inputs=st.text_input("what question do you have in mind?")

llm=OllamaLLM(model="llama3")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if inputs:
    st.write(chain.invoke({"question":inputs}))
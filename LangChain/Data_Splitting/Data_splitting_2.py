from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

#loader=TextLoader('C:\GenAi\LangChain\Data_Ingestion\speech.txt')
#text_documents=loader.load()

with open("C:\GenAi\LangChain\Data_Ingestion\speech.txt") as f:
    speech=f.read()

text_splitter=RecursiveCharacterTextSplitter(chunk_size=10,chunk_overlap=2)
output=text_splitter.create_documents([speech])
print(output)
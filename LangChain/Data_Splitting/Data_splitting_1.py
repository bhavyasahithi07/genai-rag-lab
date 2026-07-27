#recursive character text splitter, pdf loader

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader=PyPDFLoader('C:\GenAi\LangChain\Data_Ingestion\Cover_letter.pdf')
docs=loader.load()
#print(docs)
#print(type(docs))
#print(type(docs[0]))

text_splitter=RecursiveCharacterTextSplitter(chunk_size=10,chunk_overlap=5)
final_docs=text_splitter.split_documents(docs)
print(final_docs[0])


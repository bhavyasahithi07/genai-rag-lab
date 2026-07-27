#reading a pdf file

from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader('Cover_letter.pdf')
docs=loader.load()
print(docs)
print(type(docs))
print(type(docs[0]))
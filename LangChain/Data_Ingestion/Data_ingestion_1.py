#Text loader

from langchain_community.document_loaders import TextLoader

loader=TextLoader('speech.txt')

text_documents=loader.load()
print(text_documents)
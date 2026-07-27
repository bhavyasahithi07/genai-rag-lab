from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

loader=TextLoader('C:\GenAi\LangChain\Data_Ingestion\speech.txt')
text_documents=loader.load()

#if \n\n not found in chunk_size it seperates only at \n\n
#if u want seperator based on chunk size, donot use seperator
text_splitter=CharacterTextSplitter(separator="\n\n",chunk_size=5,chunk_overlap=2)
output=text_splitter.split_documents(text_documents)
print(output[0])



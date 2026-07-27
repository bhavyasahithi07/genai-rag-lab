import json
import requests
from langchain_text_splitters import RecursiveJsonSplitter

#json_data=requests.get("url").json()
json_data = {
    "name": "Bhavya",
    "age": 24,
    "skills": {
        "language": "Python",
        "framework": "LangChain"
    }
}

json_splitter=RecursiveJsonSplitter(max_chunk_size=10)
json_chunks=json_splitter.split_json(json_data)
print(json_chunks)

#.create_documents instead of split_json to get documents
#.split_text gives text as output

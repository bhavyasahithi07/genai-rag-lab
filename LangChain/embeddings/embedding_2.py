import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")

embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-V2")


text="this is a test document"
query_result = embeddings.embed_query(text)
print(query_result)
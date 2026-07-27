from pinecone import Pinecone,ServerlessSpec
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")

pc=Pinecone()




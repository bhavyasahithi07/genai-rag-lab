import os
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel,Field
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq
model=ChatGroq(model="llama-3.3-70b-versatile")

class Actor(BaseModel):
    name:str
    Role:str
    

class Movie(BaseModel):
    title:str=Field(description="title of the movie")
    rating:float=Field(description="Rating given in IMDB")
    year:int
    cast:list[Actor]
    budget:float | None=Field(None,description="budget of the movie in crores")

movie_structured=model.with_structured_output(Movie)
response=movie_structured.invoke("give me details of the movie Fidaa")
#print(response)

from typing import Annotated,TypedDict
class Actor(TypedDict):
    name:str
    Role:str

class moviedict(TypedDict):
    title:str
    rating:Annotated[float,"rating of the movie"]
    year:int
    cast:list[Actor]
    budget:float

movie_structured=model.with_structured_output(Movie)
response=movie_structured.invoke("give me details of the movie Fidaa")
print(response)


import os
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel,Field
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq
model=ChatGroq(model="llama-3.3-70b-versatile")

class Movie(BaseModel):
    title:str=Field(description="title of the movie")
    rating:float=Field(description="Rating given in IMDB")
    year:int
    budget:float | None=Field(None,description="budget of the movie in crores")

from langchain.agents import create_agent
agent=create_agent(model=model,response_format=Movie)

result=agent.invoke({
    "messages":[
    {"role":"user",
    "content":"tell me details about movie life is beautiful"}
    ]
    } )
#print(result["structured_response"])

from dataclasses import dataclass
@dataclass
class Film:
    title:str
    rating:float

agent=create_agent(model=model,response_format=Film)
result2=agent.invoke({
    "messages":[
        {
            "role":"user",
            "content":"Grab data from this- movie title=Fidaa and rating=7.5"
        }
    ]
})

print(result2["structured_response"])






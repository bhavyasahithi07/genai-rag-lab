import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

from langchain.chat_models import init_chat_model
model=init_chat_model("groq:llama-3.3-70b-versatile")
response=model.invoke("why does parrots talk?")

#for chunk in model.stream("How does parrots speak?"):
    #print(chunk.text, end="", flush=True)

response2 = model.batch([
    "why do parrots fly?",
    "what is the meaning of bhavya?",
    "future of genai rag developer oppurtunities in next 6 months?"
],
config={
    "max_concurrency":5
}
)
for response in response2:
    print(response.content)


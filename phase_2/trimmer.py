from langchain_core.messages import SystemMessage,trim_messages
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACKING_V2"]="true"


from langchain_ollama import ChatOllama
llm=ChatOllama(model="llama3")

trimmer=trim_messages(
    max_tokens=70,
    strategy='last',
    token_counter=llm,
    include_system=True,
    allow_partial=False,
    start_on="human"
)


from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
messagess=[
    SystemMessage(content="you are a good assistant"),
    HumanMessage(content="Hello"),
    AIMessage(content="Hi"),
    HumanMessage(content= "I love vanilla ice cream. do you have it"),
    AIMessage(content="I am sorry no but i can give you the recipe"),
    HumanMessage(content="thats fine i will buy outside"),
    AIMessage(content="I am sorry that i was not able to give it to u")
]

from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
prompt=ChatPromptTemplate.from_messages([
    ("system","you are a helpful assistant. Answer the questions asked"),
    MessagesPlaceholder(variable_name="messages")
])

#trimmer.invoke(messages)
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

chain=(RunnablePassthrough.assign(messages=itemgetter("messages")|trimmer)
       |prompt
       |llm)

response=chain.invoke({"messages":messagess+[HumanMessage(content="what icecream do i like?")]
             })

print(response.content)
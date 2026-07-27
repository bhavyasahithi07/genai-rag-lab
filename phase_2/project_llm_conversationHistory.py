import os
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"]="true"

#create llm
from langchain_ollama import ChatOllama
llm=ChatOllama(model="llama3")

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

#create memory storage
store={}

#Create get_session_history() function
def get_session_history(session_id:str)->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id]=ChatMessageHistory()
    return store[session_id]

#prompt template
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage,trim_messages
prompt=ChatPromptTemplate.from_messages([
    ("system","you are a helpful assistant. you should answer the questions asked in language {language}"),
    MessagesPlaceholder(variable_name="messages")
])

trimmer=trim_messages(
    max_tokens=100,
    strategy='last',
    token_counter=llm,
    include_system=True,
    allow_partial=False,
    start_on='human'

)

#chaining
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

chain=(RunnablePassthrough.assign(messages=itemgetter("messages")|trimmer)
       |prompt
       |llm)

#6. Wrap chain with RunnableWithMessageHistory
with_message_history=RunnableWithMessageHistory(chain,get_session_history,input_messages_key="messages")

#7. Create config (session_id)
config={"configurable":{"session_id":"chat1"}}

#8. Invoke the chain
with_message_history.invoke({"messages":[HumanMessage(content="hi my name is bhavya")],"language":"French"},config=config)

result=with_message_history.invoke({"messages":[HumanMessage(content="what is my name?")],"language":"french"},config=config)
print(result.content)



import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"]="true"

#step1- create llm
from langchain_ollama import ChatOllama
llm=ChatOllama(model="llama3")

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

#step2- memory storage
store={}

#step3-create function to get history
def get_session_history(session_id:str)->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id]=ChatMessageHistory()
    return store[session_id]

#with_message_history=RunnableWithMessageHistory(llm,get_session_history)




from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
#response=with_message_history.invoke([
 #   HumanMessage(content="Hi I am Bhavya. Future GenAi developer")],
  #  config=config)

#config1={"configurable":{"session_id":"chat1"}}
#response1=with_message_history.invoke([
 #   HumanMessage(content="what is my name?")
#], config=config1)

#print(response1.content)

#step4: create prompt
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
prompt=ChatPromptTemplate.from_messages([
    ("system","you are a helpful assistant. Answer the questions asked in following language {language}"),
    MessagesPlaceholder(variable_name="messages")
])

#step 5: create chain
chain=prompt|llm
#chain.invoke({"messages":[HumanMessage(content="hi my name is bhavya")]})
#this naming doesnt work cause you need to call config and with message history for it to work

#step6: add memory wrap
with_message_history=RunnableWithMessageHistory(chain,get_session_history,input_messages_key="messages")

#step7:choose conversation
config={"configurable":{"session_id":"chat1"}}

#here it works
with_message_history.invoke({"messages":[HumanMessage(content="Hi my name is bhavya")],"language":"french"},config=config)

#step 8: ask question
response=with_message_history.invoke({"messages":[HumanMessage(content="whats my name?")],"language":"french"},config=config)
print(response.content)
import streamlit as st
from pathlib import Path
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import create_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq

st.set_page_config(page_title="Langchain: Chat with SQL DB")
st.title("Langchain: Chat with SQL DB")

LOCALDB="USE_LOCALDB"
MYSQL="USE_MYSQL"

radio_opt=["Use sqllite3 database- Student.db","Connect to your MYSQL Database"]
selected_opt=st.sidebar.radio(label="choose the db you wnt to chat with",options=radio_opt)

if radio_opt.index(selected_opt)==1:
    db_uri=MYSQL
    mysql_host=st.sidebar.text_input("Provide mysql host")
    mysql_user=st.sidebar.text_input("Mysql user")
    mysql_password=st.sidebar.text_input("MYSQL password",type="password")
    mysql_db=st.sidebar.text_input("MYSQL database")
else:
    db_uri=LOCALDB

api_key=st.sidebar.text_input(label="Groq Api Key",type="password")

if not db_uri:
    st.info("please enter db info")
    st.stop()

if not api_key:
    st.info("please add the api key")
    st.stop()

llm=ChatGroq(model="openai/gpt-oss-20b",groq_api_key=api_key)

@st.cache_resource(ttl=7200)
def configure_db(db_uri,mysql_host=None,mysql_user=None,mysql_password=None,mysql_db=None):
    if db_uri==LOCALDB:
        dbfilepath=(Path(__file__).parent/"student.db").absolute()
        print(dbfilepath)
        creator=lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro",uri=True)
        return SQLDatabase(create_engine("sqlite:///",creator=creator))

    elif db_uri==MYSQL:
        if not (mysql_host and mysql_db and mysql_password and mysql_user):
            st.error("please provide connection details")
            st.stop()

        return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))
    
if db_uri==MYSQL:
    db=configure_db(db_uri,mysql_host,mysql_user,mysql_password,mysql_db)
else:
    db=configure_db(db_uri)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

tools = toolkit.get_tools()
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
You are a SQL database assistant.

You MUST use the available SQL tools to answer database questions.

For questions about data:
1. First inspect the available tables if needed.
2. Use the SQL tools to query the database.
3. Return the actual database results.
4. Never make up database information.
"""
)

if "messages" not in st.session_state:
    st.session_state["messages"]=[{
        "role":"assistant","content":"How can i help u?"
    }]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query=st.chat_input(placeholder="ask me anything?...")

if user_query:
    st.session_state.messages.append({"role":"user","content":user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": user_query
            }]},config={"callbacks": [streamlit_callback]})
        answer=response["messages"][-1].content
        st.session_state.messages.append({"role":"assistant","content":answer})
        st.write(answer)

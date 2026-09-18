#PROJECTS\5-search_engine_Tools_agents.py
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import create_agent
#from langchain.agents import initialize_agent,AgentType
from langchain_community.callbacks import StreamlitCallbackHandler
#from langchain.callbacks import StreamlitCallbackHandler

import os
from dotenv import load_dotenv

arxiv_wrapper=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(
    name="duckduckgo_search",
    description="Search the web for current and recent information."
)

st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Groq API Key:",type="password")

st.title("Langchain- chat with search")

if "messages" not in st.session_state:
    st.session_state["messages"]=[{
            "role":"assistant","content":"Hi, I'm a chatbot who can search the web. How can I assist you"
        }]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt:=st.chat_input(placeholder="HELLO TYPE ANYTHING...."):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm=ChatGroq(groq_api_key=api_key,model="openai/gpt-oss-20b",temperature=0)
    tools=[search,arxiv,wiki]

    search_agent=create_agent(model=llm,tools=tools,system_prompt="You are a helpful AI assistant that can search the web.")

    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=search_agent.invoke({"messages":st.session_state.messages},
                                     config={"callbacks":[st_cb]})
        answer=response["messages"][-1].content
        st.session_state.messages.append({'role':'assistant',"content":answer})
        st.write(answer)
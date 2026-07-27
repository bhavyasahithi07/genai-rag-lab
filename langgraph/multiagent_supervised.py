from typing import Literal
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage,SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph,START,END,MessagesState
from langgraph.prebuilt import create_react_agent,ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"]="true"

llm=ChatGroq(model="llama-3.3-70b-versatile")


class SupervisorState(MessagesState):
    next_agent:str=""   #safer
    research_data:str=""
    analysis_data:str=""
    final_data:str=""
    complete_task:bool=False
    current_task:str=""

from langchain_core.prompts import ChatPromptTemplate
def create_supervisor_agent():
    """creates the supervisor decision chain"""
    supervisor_prompt=ChatPromptTemplate.from_messages([
        ("system","""You are a supervisor who assign tasks to three agents based on necessity
         Agents are:
         1.Researcher:Gathers data
         2.Analyst:analyses the data
         3.Writer: summarises data and write reports

         based on current state and conversation decide which agent to work next, if total work is done respond with "done"
         current state:
         Has research data:{has_research}
         Has analysis:{has_analysis}
         Has report:{has_report}

         Respond with agent name or "done"
         """),("human","{task}")
         
         ])
    
    return supervisor_prompt|llm

def supervisor_agent(state:SupervisorState)->dict:
    """supervisor_agent decides next agent using Groq LLM"""
    task = state.get("current_task")

    if not task:
        messages = state["messages"]
        task = messages[0].content if messages else "No task"

    has_research=bool(state.get("research_data",""))
    has_analysis=bool(state.get("analysis_data",""))
    has_report=bool(state.get("final_data",""))

    chain=create_supervisor_agent()
    decision=chain.invoke(
        {
            "task":task,
            "has_research":has_research,
            "has_analysis":has_analysis,
            "has_report":has_report
        }
    )

    decision_text=decision.content.strip().lower()
    print(f"decision text:{decision_text}")

    if "done" in decision_text or has_report:
        next_agent="end"
        supervisor_msg="All tasks completed"
    elif "researcher" in decision_text or not has_research:
        next_agent="researcher"
        supervisor_msg="lets start with researcher"
    elif "analyst" in decision_text or not (has_research and has_analysis):
        next_agent="analyst"
        supervisor_msg="go to analyst"
    elif "writer" in decision_text or not (has_analysis and has_report):
        next_agent="writer"
        supervisor_msg="go to writer"    
    else:
        next_agent="end"
        supervisor_msg="all done"

    return {
        "messages":[AIMessage(content=supervisor_msg)],
        "next_agent":next_agent,
        "current_task":task
    }
    
    
  
def research_agent(state:SupervisorState)->dict:
    """Researcher uses groq to gather data"""
    task=state.get("current_task","research topic")
    research_prompt=f"""as a reseach specialist provide information about {task}
    Include the following:
    1.key facts
    2.current trends
    3.real world examples 
    Be concise but thorough"""

    research_response=llm.invoke([HumanMessage(content=research_prompt)])
    research_data=research_response.content

    agent_message=f"I have completed {task}\n the findings are {research_data}"

    return {
        "messages":[AIMessage(content=agent_message)],
        "research_data":research_data,
        "next_agent":"supervisor"
    }

def analyst_agent(state:SupervisorState)->dict:
    """analyst uses groq to analyse data"""
    research_data=state.get("research_data","")
    task=state.get("current_task","")

    analysis_prompt=f"""As a analyst analyse data and provide insights
    Research data:{research_data}

    provide:
    1.key insights
    2.risks and opputunities
    3.recommendations
    focus on actionable insights related to {task}  """

    analysis_response=llm.invoke([HumanMessage(content=analysis_prompt)])
    analysis_data=analysis_response.content

    agent_message=f"I have completed {task}\n the findings are {analysis_data[:400]}"

    return {
        "messages":[AIMessage(content=agent_message)],
        "analysis_data":analysis_data,
        "next_agent":"supervisor"
    }

def writer_agent(state:SupervisorState)->dict:
    """Writer uses groq to create final report"""
    research_data=state.get("research_data","")
    analysis_data=state.get("analysis_data","")
    task=state.get("current_task","")

    writing_prompt=f"""As a professional writer create a report based on:
    task:{task}
    research findings:{research_data[:1000]}
    Analysis:{analysis_data[:1000]}
    create a well structured report with 
    1.summary
    2.key findings
    3.Analysis and insights
    4.recommendations
    5.conclusion
    keep it professional and concise"""

    report_response=llm.invoke([HumanMessage(content=writing_prompt)])
    report=report_response.content


    final_data=f"""FINAL REPORT
    Report compiled by multi agent Ai system powered by groq
    {report}
    """

    return {
        "messages":[AIMessage(content=f"writer: Report dONE")],
        "final_data":final_data,
        "next_agent":"supervisor",
        "complete_task":True
    }

def router(state:SupervisorState)->Literal["supervisor","researcher","analyst","writer"]:
    """Routes to next agent based on state"""
    next_agent=state.get("next_agent","supervisor")

    if next_agent=="end" or state.get("complete_task",False):
        return END
    if next_agent in ["supervisor","researcher","analyst","writer"]:
        return next_agent
    return "supervisor"

workflow=StateGraph(SupervisorState)

workflow.add_node("supervisor",supervisor_agent)
workflow.add_node("researcher",research_agent)
workflow.add_node("analyst",analyst_agent)
workflow.add_node("writer",writer_agent)

workflow.set_entry_point("supervisor")

for node in ["supervisor","researcher","analyst","writer"]:
    workflow.add_conditional_edges(
        node,
        router,
        {
            "supervisor":"supervisor",
            "researcher":"researcher",
            "analyst":"analyst",
            "writer":"writer",
            END:END
        }
    )

graph=workflow.compile()

response = graph.invoke({
    "messages": [
        HumanMessage(content="what are benefits and risks of AI in healthcare")
    ]
})
print(response)


    
    
    
  

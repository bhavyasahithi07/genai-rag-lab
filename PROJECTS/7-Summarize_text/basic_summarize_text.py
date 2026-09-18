import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
api_key=os.getenv("GROQ_API_KEY")
llm=ChatGroq(groq_api_key=api_key,model="openai/gpt-oss-20b")

from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
speech="""# Founder Speech — Building the Future of AI

When we started building Claude, we weren't simply trying to create another AI product.

We wanted to answer a much bigger question:

**Can we build AI that is incredibly capable, genuinely useful, and worthy of people's trust?**

AI has the potential to change almost every part of our lives. It can help someone learn a new subject, help a developer build software, help researchers solve difficult problems, and give people access to capabilities that were previously available only to large organizations.

But capability alone isn't enough.

The more powerful AI becomes, the more carefully we have to think about how it behaves.

We need systems that can recognize uncertainty, avoid harmful behavior, follow human instructions, and remain reliable even when situations become complicated.

That's why safety cannot be something we add at the end.

It has to be part of how we build AI from the beginning.

At the same time, we shouldn't be afraid of progress.

The opportunity in front of us is enormous.

Imagine a student having an intelligent tutor available whenever they need one.

Imagine a scientist having an AI research partner that can help explore thousands of ideas.

Imagine a programmer turning an idea into a working application in hours instead of weeks.

Imagine small businesses having access to capabilities that once required entire teams.

These possibilities are why we build.

But our responsibility is to make sure that as AI becomes more powerful, it also becomes more trustworthy.

We don't want AI simply because it is impressive.

We want AI because it can help people accomplish things they couldn't accomplish before.

The future of AI shouldn't be about humans versus machines.

It should be about humans **with** machines—people using increasingly powerful tools to learn faster, create more, discover more, and solve problems that once seemed impossible.

There will be mistakes. There will be difficult questions. There will be risks we haven't anticipated yet.

Our job is not to pretend those challenges don't exist.

Our job is to face them directly, learn from them, and keep improving.

If we do that, AI can become more than a technological breakthrough.

It can become one of the most powerful tools humanity has ever created.

And that is the future we want to build.

Thank you.

"""
#==============================================================================================================
chat_message=[
    SystemMessage(content="You are a expert in summarizing speeches"),
    HumanMessage(content=f"please provide a short and concise summary of this speech: {speech}")
]

output=llm.get_num_tokens(speech)
print(output)

print(llm.invoke(chat_message).content)

#====================================================================================
from langchain_core.prompts import PromptTemplate

generictemplate="""
Write a summary of this {speech}
and translate it ti {language}
"""

prompt=PromptTemplate(
    input_variables=['speech','language'],
    template=generictemplate
)

chain=prompt | llm

response=chain.invoke({"speech":speech,"language":"Telugu"})
print(response.content)
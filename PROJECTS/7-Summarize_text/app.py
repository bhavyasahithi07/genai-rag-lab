import validators,streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader


st.set_page_config(page_title="Langchain:Summarize text from yt or website url")
st.title("Summarizing YT or WEBSITE URL")

with st.sidebar:
    groq_api_key=st.text_input("GROQ API KEY",value="",type="password")

generic_url=st.text_input("URL",label_visibility="visible")


prompt_template="""
provide a summary of the content in 100 words
content:{context}
"""

prompt=PromptTemplate(template=prompt_template,input_variables=["context"])

if st.button("Summarize context"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
        st.stop()

    elif not validators.url(generic_url):
        st.error("Please provide a valid url")
        st.stop()

    else:
        try:
            llm=ChatGroq(model="openai/gpt-oss-20b",groq_api_key=groq_api_key)
            with st.spinner("waiting..."):
                if "youtube.com" in generic_url:
                    loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=False)
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],
                                                 ssl_verify=False,
                                                 headers={"User-Agent": (
                                                     "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                                     "AppleWebKit/537.36 (KHTML, like Gecko) "
                                                     "Chrome/131.0.0.0 Safari/537.36")})

                docs=loader.load()
                chain=create_stuff_documents_chain(llm=llm,prompt=prompt)
                output=chain.invoke({"context":docs})
                st.success(output)

        except Exception as e:
                st.exception(e)
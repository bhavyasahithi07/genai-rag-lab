#web based loader
from langchain_community.document_loaders import WebBaseLoader
import bs4

loader=WebBaseLoader(web_path="https://arxiv.org/abs/1706.03762",
                     bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_="abstract mathjax")))


docs=loader.load()
print(docs)
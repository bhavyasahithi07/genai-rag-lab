from langchain_community.document_loaders import PyPDFLoader
import re
from pathlib import Path
loader=PyPDFLoader(r"summary_of_benefits.pdf")
documents=loader.load()

print(documents)

file_path=Path("summary_of_benefits.pdf")
# print("pages:",len(documents))
# valid_documents=[]
# for doc in documents:
#     if doc.page_content.strip()=="":
#         continue
#     valid_documents.append(doc)

# print("number of documents:",len(documents))
# print("number of valid documents:",len(valid_documents))

# print("----------------------BEFORE:")
# print(valid_documents[2].page_content)
# print(valid_documents[2].metadata)

# #cleaning
# cleaned_documents=[]
# for doc in valid_documents:
#     text=doc.page_content
#     text=text.replace("For more information about limitations and exceptions, see the plan or policy document at www.netbenefits.com","")
#     text=re.sub(r"Page \d+ of \d+", "", text)
#     text=re.sub(r"[ \t]+"," ",text)
#     text=text.replace("\x00"," ")
#     text=text.strip()

#     doc.page_content=text
#     cleaned_documents.append(doc)

# print("\n---------------------AFTER:")
# print(cleaned_documents[2].page_content)
# print(cleaned_documents[2].metadata)

# for doc in cleaned_documents:
#     doc.metadata["document_type"]="Health Benefits"
#     doc.metadata["category"]="Insurance"
#     doc.metadata["year"]=2026
#     doc.metadata["file_name"]=file_path.name
#     doc.metadata["type"]=file_path.suffix

# print(cleaned_documents[0].metadata)
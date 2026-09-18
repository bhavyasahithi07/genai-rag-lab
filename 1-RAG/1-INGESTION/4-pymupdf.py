import fitz

pdf = fitz.open("summary_of_benefits.pdf")

for page_num, page in enumerate(pdf):
    text = page.get_text()

    print("PAGE:", page_num + 1)
    print(text)
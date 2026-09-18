import pdfplumber
#poor result on this PDF
with pdfplumber.open("Health_plan.pdf") as pdf:
    for page_num, page in enumerate(pdf.pages):
        print("PAGE:", page_num + 1)

        # Extract normal text
        text = page.extract_text()
        print("TEXT:")
        print(text)

        # Extract tables
        tables = page.extract_tables()

        print("TABLES:")
        for table in tables:
            for row in table:
                print(row)
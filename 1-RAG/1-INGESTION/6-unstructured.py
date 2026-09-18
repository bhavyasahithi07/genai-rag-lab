from unstructured.partition.pdf import partition_pdf

elements = partition_pdf(
    filename="Health_plan.pdf"
)

for element in elements:
    print(type(element).__name__)
    print(element.text)
    print("-----")
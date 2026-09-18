from docling.document_converter import DocumentConverter
import re

converter = DocumentConverter()

result = converter.convert("Health_plan.pdf")

document = result.document

# Extract from Docling
markdown_output = document.export_to_markdown()
json_output = document.export_to_dict()


# ---------------- CLEANING ----------------

markdown_output = markdown_output.replace("\x00", " ")

# Remove extra spaces/tabs
markdown_output = re.sub(r"[ \t]+", " ", markdown_output)

# Remove too many blank lines
markdown_output = re.sub(r"\n{3,}", "\n\n", markdown_output)

# Remove whitespace at beginning/end
markdown_output = markdown_output.strip()


# ---------------- AFTER CLEANING ----------------

print(markdown_output)
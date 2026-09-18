from docling.document_converter import DocumentConverter,PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

pipeline_options=PdfPipelineOptions()
pipeline_options.do_ocr=True
pipeline_options.ocr_options.force_full_page_ocr = True

converter=DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options
        )
    }
)

result=converter.convert("isss.pdf")
print(result.document.export_to_markdown())
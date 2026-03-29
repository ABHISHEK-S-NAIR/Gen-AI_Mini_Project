from PyPDF2 import PdfReader


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    from io import BytesIO

    reader = PdfReader(BytesIO(pdf_bytes))

    pages_text: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages_text.append(page_text)

    return "\n".join(pages_text).strip()

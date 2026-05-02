from io import BytesIO
import re

from PyPDF2 import PdfReader


def _normalize_cell(cell: object) -> str:
    if cell is None:
        return ""
    text = str(cell)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _table_to_text(rows: list[list[str]]) -> str:
    if not rows:
        return ""

    max_cols = max(len(row) for row in rows)
    normalized = [row + [""] * (max_cols - len(row)) for row in rows]
    lines = [" | ".join(cell for cell in row).strip() for row in normalized]
    return "\n".join(lines).strip()


def _format_tables_for_context(tables: list[dict[str, object]]) -> str:
    if not tables:
        return ""

    blocks = []
    for table in tables:
        page = table.get("page", "?")
        index = table.get("table_index", "?")
        text = table.get("text") or ""
        if not text and "rows" in table:
            text = _table_to_text(table.get("rows", []))
        if not text:
            continue
        blocks.append(f"Table {index} (page {page})\n{text}")

    if not blocks:
        return ""

    return "[TABLES]\n" + "\n\n".join(blocks)


def _extract_with_pdfplumber(pdf_bytes: bytes) -> tuple[str, list[dict[str, object]], list[dict[str, object]]]:
    import pdfplumber

    text_pages: list[str] = []
    tables: list[dict[str, object]] = []
    figures: list[dict[str, object]] = []

    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for page_index, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            if page_text:
                text_pages.append(page_text)

            try:
                page_tables = page.extract_tables() or []
            except Exception:
                page_tables = []

            for table_index, table in enumerate(page_tables, start=1):
                if not table:
                    continue
                normalized = [
                    [_normalize_cell(cell) for cell in (row or [])]
                    for row in table
                ]
                normalized = [row for row in normalized if any(cell for cell in row)]
                if not normalized:
                    continue

                tables.append(
                    {
                        "page": page_index + 1,
                        "table_index": table_index,
                        "rows": normalized,
                        "text": _table_to_text(normalized),
                        "source": "pdfplumber",
                    }
                )

            for figure_index, image in enumerate(page.images or [], start=1):
                bbox = None
                if isinstance(image, dict):
                    x0 = image.get("x0")
                    x1 = image.get("x1")
                    top = image.get("top")
                    bottom = image.get("bottom")
                    if all(v is not None for v in (x0, top, x1, bottom)):
                        bbox = [x0, top, x1, bottom]

                figures.append(
                    {
                        "page": page_index + 1,
                        "figure_index": figure_index,
                        "bbox": bbox,
                        "source": "pdfplumber",
                    }
                )

    return "\n".join(text_pages).strip(), tables, figures


def _extract_with_pypdf2(pdf_bytes: bytes) -> tuple[str, list[dict[str, object]], list[dict[str, object]]]:
    reader = PdfReader(BytesIO(pdf_bytes))
    pages_text: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages_text.append(page_text)
    return "\n".join(pages_text).strip(), [], []


def extract_pdf_content(
    pdf_bytes: bytes,
) -> tuple[str, list[dict[str, object]], list[dict[str, object]]]:
    try:
        text, tables, figures = _extract_with_pdfplumber(pdf_bytes)
        if text or tables or figures:
            return text, tables, figures
    except Exception:
        pass

    return _extract_with_pypdf2(pdf_bytes)


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    text, _tables, _figures = extract_pdf_content(pdf_bytes)
    return text


def build_tables_text(tables: list[dict[str, object]]) -> str:
    return _format_tables_for_context(tables)

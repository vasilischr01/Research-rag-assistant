from dataclasses import dataclass
from pathlib import Path

import fitz


@dataclass(frozen=True)
class PageText:
    document: str
    page: int
    text: str

def extract_pdf_pages(path: Path) -> list[PageText]:
    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")
    pages = []
    with fitz.open(path) as pdf:
        for i, page in enumerate(pdf):
            text = page.get_text("text").strip()
            if text:
                pages.append(PageText(path.name, i + 1, text))
    if not pages:
        raise ValueError("The PDF contains no extractable text.")
    return pages

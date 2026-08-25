from dataclasses import dataclass

from src.core.pdf import PageText


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document: str
    page: int
    text: str

def chunk_pages(pages: list[PageText], chunk_size: int, overlap: int) -> list[Chunk]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk_size/overlap.")
    chunks = []
    step = chunk_size - overlap
    for page in pages:
        text = " ".join(page.text.split())
        start, n = 0, 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(f"{page.document}:p{page.page}:c{n}", page.document, page.page, chunk_text))
            if end >= len(text):
                break
            start += step
            n += 1
    return chunks

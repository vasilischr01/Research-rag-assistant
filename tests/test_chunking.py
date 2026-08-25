import pytest

from src.core.chunking import chunk_pages
from src.core.pdf import PageText


def test_chunking_preserves_metadata():
    chunks = chunk_pages([PageText("paper.pdf", 3, "A" * 120)], 50, 10)
    assert len(chunks) == 3
    assert chunks[0].page == 3
    assert chunks[0].chunk_id == "paper.pdf:p3:c0"

def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_pages([PageText("a.pdf", 1, "hello")], 20, 20)

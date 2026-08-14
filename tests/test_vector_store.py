import numpy as np
from src.core.chunking import Chunk
from src.core.vector_store import VectorStore

def test_vector_store_returns_best_match(tmp_path):
    store = VectorStore(tmp_path / "index.faiss", tmp_path / "metadata.json")
    chunks = [Chunk("a","a.pdf",1,"machine learning"), Chunk("b","b.pdf",2,"vision")]
    store.add(chunks, np.asarray([[1.,0.],[0.,1.]], dtype="float32"))
    hits = store.search(np.asarray([1.,0.], dtype="float32"), 1)
    assert hits[0]["chunk_id"] == "a"

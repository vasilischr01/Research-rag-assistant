import json
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from src.core.chunking import Chunk


class VectorStore:
    def __init__(self, index_path=Path("data/indexes/index.faiss"), metadata_path=Path("data/indexes/metadata.json")):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.chunks = []

    @property
    def size(self):
        return len(self.chunks)

    def load(self):
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        if self.metadata_path.exists():
            self.chunks = [Chunk(**x) for x in json.loads(self.metadata_path.read_text(encoding="utf-8"))]

    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
        self.metadata_path.write_text(json.dumps([asdict(c) for c in self.chunks], indent=2), encoding="utf-8")

    def add(self, chunks, embeddings):
        if not chunks:
            return
        embeddings = np.asarray(embeddings, dtype="float32")
        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
            raise ValueError("Invalid embedding matrix.")
        if self.index is None:
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
        if self.index.d != embeddings.shape[1]:
            raise ValueError("Embedding dimension mismatch.")
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        self.save()

    def search(self, query_embedding, top_k):
        if self.index is None or not self.chunks:
            return []
        q = np.asarray(query_embedding, dtype="float32")
        if q.ndim == 1:
            q = q.reshape(1, -1)
        scores, ids = self.index.search(q, min(top_k, len(self.chunks)))
        out = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            c = self.chunks[int(idx)]
            out.append({"chunk_id": c.chunk_id, "document": c.document, "page": c.page, "text": c.text, "score": float(score)})
        return out

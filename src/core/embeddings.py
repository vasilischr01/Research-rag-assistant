from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from src.core.config import settings


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(
        settings.embedding_model
    )


def embed_texts(
    texts: list[str],
) -> np.ndarray:
    if not texts:
        return np.empty(
            (0, 0),
            dtype="float32",
        )

    embeddings = get_embedding_model().encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return np.asarray(
        embeddings,
        dtype="float32",
    )

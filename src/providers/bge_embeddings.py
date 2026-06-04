"""
BGE-small-en-v1.5 embedding provider implementation.

Model: BAAI/bge-small-en-v1.5
Dimensions: 384
Disk size: ~133 MB
Runtime RAM: ~300-600 MB (CPU-only)
Normalize: L2-normalized vectors (recommended for cosine similarity)
"""

import logging
from typing import List

from sentence_transformers import SentenceTransformer

from src.providers.interfaces import IEmbeddingProvider

logger = logging.getLogger(__name__)


class BGEEmbeddingProvider(IEmbeddingProvider):
    """
    Local CPU-only embedding provider using BAAI/bge-small-en-v1.5.

    Produces 384-dimensional L2-normalized vectors suitable for cosine
    similarity search in ChromaDB.

    The model is loaded once at startup and reused for all requests.
    """

    EMBEDDING_DIM = 384

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        logger.info(f"Loading embedding model: {model_name} ...")
        self._model = SentenceTransformer(model_name)
        logger.info(f"Embedding model loaded (dim={self.EMBEDDING_DIM})")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of document texts.
        normalize_embeddings=True produces unit-length vectors for cosine similarity.
        """
        if not texts:
            return []
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 50,  # Progress bar only for large batches
            batch_size=32,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string.
        Returns a 1D list of 384 floats.
        """
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

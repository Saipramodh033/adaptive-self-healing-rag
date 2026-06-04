"""
ChromaDB persistent vector store implementation.

Uses raw chromadb (not langchain-chroma) for full control over:
- Embedding injection (pre-computed vectors, no re-embedding)
- Query result structure (explicit include= for distances)
- Document count health checks

ChromaDB API notes (validated):
- PersistentClient(path=...) — current API, stable since v0.5
- get_or_create_collection() — safe for re-runs (no error if exists)
- collection.query() returns list-of-lists; use [0] for single-query results
- Default include omits distances; must be explicit
"""

import logging
import uuid
from typing import List

import chromadb
from langchain_core.documents import Document

from src.providers.interfaces import IEmbeddingProvider, IVectorStore

logger = logging.getLogger(__name__)


class ChromaVectorStore(IVectorStore):
    """
    Persistent ChromaDB vector store with externally-provided embeddings.

    Documents are stored with pre-computed BGE-small embeddings.
    All 384-dim vectors must be consistent — ChromaDB enforces dimension
    consistency after the first insert.
    """

    def __init__(
        self,
        persist_dir: str,
        collection_name: str,
        embedding: IEmbeddingProvider,
    ):
        logger.info(f"Connecting to ChromaDB at: {persist_dir}")
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine distance metric
        )
        self._embedding = embedding
        count = self._collection.count()
        logger.info(
            f"ChromaDB ready — collection='{collection_name}', documents={count}"
        )

    def add_documents(self, documents: List[Document]) -> None:
        """
        Store LangChain Documents with pre-computed embeddings.
        Generates stable UUID-based IDs from document content.
        """
        if not documents:
            return

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # Compute embeddings in batch
        embeddings = self._embedding.embed_documents(texts)

        # Generate deterministic IDs (prevents duplicates on re-ingestion)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, text[:200])) for text in texts]

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info(f"Added {len(documents)} documents to ChromaDB")

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Return top-k documents most similar to the query.

        Uses explicit include= to retrieve distances for observability.
        Results include similarity distance in document metadata.
        """
        query_embedding = self._embedding.embed_query(query)

        # Explicit include= required — default omits distances
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self._collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )

        # Results are list-of-lists (one inner list per query)
        docs = results["documents"][0]        # List[str]
        metas = results["metadatas"][0]       # List[dict]
        distances = results["distances"][0]   # List[float] (lower = more similar)

        return [
            Document(
                page_content=doc,
                metadata={**meta, "similarity_distance": round(dist, 4)},
            )
            for doc, meta, dist in zip(docs, metas, distances)
        ]

    def get_document_count(self) -> int:
        """Return total number of stored document chunks."""
        return self._collection.count()

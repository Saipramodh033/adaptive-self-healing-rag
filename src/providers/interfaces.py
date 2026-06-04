"""
Abstract provider interfaces — the contracts that decouple every layer.

This is the most important file in the project.
All graph nodes depend ONLY on these interfaces, never on concrete implementations.

Swapping providers:
  Groq → OpenAI:     write OpenAILLMProvider(ILLMProvider)    in groq_llm.py
  ChromaDB → Pinecone: write PineconeStore(IVectorStore)      in chroma_store.py
  BGE → OpenAI embeds: write OpenAIEmbedding(IEmbeddingProvider) in bge_embeddings.py

In all cases: zero changes to any graph node.
"""

from abc import ABC, abstractmethod
from typing import List, AsyncGenerator

from langchain_core.documents import Document


class ILLMProvider(ABC):
    """
    Contract for any LLM backend (Groq, OpenAI, Ollama, etc.)

    Two model tiers:
    - invoke_json()  → always fast model (8B) — routing, grading, fact-checking
    - invoke_text()  → fast by default; use_power=True for final synthesis (70B)
    - astream_text() → power model for real-time streaming
    """

    @abstractmethod
    def invoke_json(self, prompt: str, system: str) -> dict:
        """
        Send a prompt expecting a structured JSON response.
        Used by: RouterNode, DocGraderNode, HallucinationGraderNode.
        Always uses the fast model (8B) for low-latency structured tasks.
        """
        ...

    @abstractmethod
    def invoke_text(self, prompt: str, system: str, *, use_power: bool = False) -> str:
        """
        Send a prompt expecting a free-text response.
        Used by: GeneratorNode (use_power=True), QueryRewriterNode, DirectResponderNode.
        Falls back to fast model if 70B daily budget is exhausted.
        """
        ...

    @abstractmethod
    async def astream_text(self, prompt: str, system: str) -> AsyncGenerator[str, None]:
        """
        Async generator yielding text chunks for real-time streaming.
        Used by: FastAPI /chat/stream endpoint.
        Uses power model (70B) with budget guard.
        """
        ...


class IEmbeddingProvider(ABC):
    """
    Contract for any embedding model.
    Implementations must produce vectors of consistent dimensionality.
    """

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts. Used during document ingestion.
        Returns list of float vectors.
        """
        ...

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string. Used during retrieval.
        Returns a single float vector.
        """
        ...


class IVectorStore(ABC):
    """
    Contract for any vector database.
    Implementations must support document storage and similarity search.
    """

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """
        Store documents with pre-computed embeddings.
        Used by: scripts/ingest.py during initial knowledge base loading.
        """
        ...

    @abstractmethod
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Return top-k documents most similar to the query string.
        Used by: RetrieverNode during graph execution.
        """
        ...

    @abstractmethod
    def get_document_count(self) -> int:
        """
        Return total number of documents stored.
        Used by: FastAPI /health endpoint.
        """
        ...

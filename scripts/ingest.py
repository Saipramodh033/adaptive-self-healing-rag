"""
Knowledge base ingestion script — loads ShopEase docs into ChromaDB.

Run this ONCE before starting the FastAPI server:
    python scripts/ingest.py

What this script does:
1. Loads all Markdown files from data/knowledge_base/ (recursive)
2. Adds a 'source' metadata field (relative file path) to each document
3. Splits documents into overlapping chunks for better retrieval
4. Embeds each chunk using BGE-small-en-v1.5 (local, free, 384-dim)
5. Stores chunks in a persistent ChromaDB collection

Re-running this script is safe:
- ChromaDB uses content-hash IDs (UUID5) — exact duplicate chunks are skipped
- Running again after adding new documents adds only the new chunks
- Running again after editing documents creates new chunks (old ones remain)
  To do a clean re-ingest: delete the chroma_db/ directory first

Design notes:
- Uses TextLoader (not UnstructuredMarkdownLoader) for simplicity — Markdown
  headers, tables, and lists are preserved as plain text, which works well
  for RAG retrieval without needing markdown parsing.
- chunk_size and chunk_overlap come from Settings to keep config centralised.
- RecursiveCharacterTextSplitter splits on paragraph → sentence → word
  boundaries in that preference order, which respects document structure.
"""

import logging
import sys
from pathlib import Path

# ── Path setup ─────────────────────────────────────────────────────────────────
# Add the project root to sys.path so we can import src.* modules.
# This is needed because scripts/ is NOT a Python package.
# Alternative: run with `python -m scripts.ingest` from the project root.
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from src.config import load_settings
from src.providers.bge_embeddings import BGEEmbeddingProvider
from src.providers.chroma_store import ChromaVectorStore

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Document loader ────────────────────────────────────────────────────────────

def load_documents(knowledge_base_dir: str):
    """
    Recursively load all .md and .txt files from the knowledge base directory.

    Adds 'source' metadata to each document (relative path from project root).
    This metadata is stored in ChromaDB and used in thought-trace logs so
    the RAG pipeline can tell you which document was retrieved.

    Args:
        knowledge_base_dir: Path to the knowledge base folder (relative to CWD)

    Returns:
        List of LangChain Document objects
    """
    kb_path = Path(knowledge_base_dir)
    if not kb_path.exists():
        logger.error(f"Knowledge base directory not found: {kb_path.resolve()}")
        sys.exit(1)

    all_docs = []
    file_patterns = ["**/*.md", "**/*.txt"]

    for pattern in file_patterns:
        for file_path in sorted(kb_path.glob(pattern)):
            try:
                loader = TextLoader(str(file_path), encoding="utf-8")
                docs = loader.load()

                # Add human-readable source metadata (relative path)
                # Use resolve() to make absolute before relative_to() — required on Windows
                relative_path = file_path.resolve().relative_to(PROJECT_ROOT.resolve())
                for doc in docs:
                    doc.metadata["source"] = str(relative_path).replace("\\", "/")
                    doc.metadata["category"] = file_path.parent.name  # e.g. "policies", "faqs"
                    doc.metadata["filename"] = file_path.stem         # e.g. "refund_and_returns"

                all_docs.extend(docs)
                logger.info(f"  Loaded: {relative_path} ({len(docs)} document(s))")

            except Exception as e:
                logger.warning(f"  Skipped: {file_path} — {e}")

    return all_docs


# ── Ingestion ──────────────────────────────────────────────────────────────────

def ingest():
    """
    Main ingestion function — loads, splits, embeds, and stores documents.

    Process:
    1. Load settings from .env
    2. Initialise providers (embedding + ChromaDB)
    3. Load all knowledge base documents
    4. Split into chunks (with overlap for context continuity)
    5. Store in ChromaDB with source metadata
    """
    logger.info("=" * 60)
    logger.info("ShopEase Knowledge Base Ingestion")
    logger.info("=" * 60)

    # ── Load settings ──────────────────────────────────────────────────────────
    settings = load_settings()
    logger.info(f"Embedding model : {settings.embedding_model}")
    logger.info(f"Chunk size      : {settings.chunk_size} chars")
    logger.info(f"Chunk overlap   : {settings.chunk_overlap} chars")
    logger.info(f"ChromaDB path   : {settings.chroma_persist_dir}")
    logger.info(f"Collection name : {settings.chroma_collection_name}")

    # ── Initialise providers ───────────────────────────────────────────────────
    logger.info("\n[1/4] Loading embedding model ...")
    embedding = BGEEmbeddingProvider(model_name=settings.embedding_model)

    logger.info("[2/4] Connecting to ChromaDB ...")
    store = ChromaVectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
        embedding=embedding,
    )
    existing_count = store.get_document_count()
    logger.info(f"       Existing chunks in collection: {existing_count}")

    # ── Load documents ─────────────────────────────────────────────────────────
    logger.info("\n[3/4] Loading knowledge base documents ...")
    kb_dir = "data/knowledge_base"
    documents = load_documents(kb_dir)
    logger.info(f"\nTotal documents loaded: {len(documents)}")

    if not documents:
        logger.error("No documents found. Check data/knowledge_base/ directory.")
        sys.exit(1)

    # ── Split into chunks ──────────────────────────────────────────────────────
    logger.info("\n[4/4] Splitting documents into chunks ...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        # Split preference order: paragraphs → sentences → words → characters
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Total chunks created: {len(chunks)}")

    if chunks:
        avg_len = sum(len(c.page_content) for c in chunks) / len(chunks)
        logger.info(f"Average chunk length: {avg_len:.0f} chars")

    # ── Store in ChromaDB ──────────────────────────────────────────────────────
    logger.info("\nStoring chunks in ChromaDB ...")
    store.add_documents(chunks)

    # ── Summary ────────────────────────────────────────────────────────────────
    final_count = store.get_document_count()
    new_chunks = final_count - existing_count

    logger.info("\n" + "=" * 60)
    logger.info("Ingestion Complete!")
    logger.info("=" * 60)
    logger.info(f"  Source documents : {len(documents)}")
    logger.info(f"  Chunks created   : {len(chunks)}")
    logger.info(f"  New chunks added : {new_chunks}")
    logger.info(f"  Total in ChromaDB: {final_count}")
    logger.info("\nNext steps:")
    logger.info("  1. Start the API:  uvicorn src.api.app:app --reload --port 8000")
    logger.info("  2. Start the UI:   chainlit run src/ui/app.py --port 8080")
    logger.info("  3. Test health:    curl http://localhost:8000/health")
    logger.info("=" * 60)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ingest()

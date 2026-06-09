import asyncio
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from src.config import load_settings
from src.core.graph_builder import build_graph
from src.providers.bge_embeddings import BGEEmbeddingProvider
from src.providers.chroma_store import ChromaVectorStore
from src.providers.groq_llm import GroqLLMProvider

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

QUERIES = [
    {
        "desc": "Unanswerable / Missing Info",
        "q": "Do you sell gaming consoles?",
        "expected_route": "rag -> query_rewriter -> escalate"
    },
    {
        "desc": "Conceptual Matching",
        "q": "My laptop stopped working after 6 months, what do I do?",
        "expected_route": "rag -> generator"
    }
]

async def main():
    settings = load_settings()
    llm = GroqLLMProvider(settings=settings)
    embeddings = BGEEmbeddingProvider(model_name=settings.embedding_model)
    vectorstore = ChromaVectorStore(
        collection_name=settings.chroma_collection_name,
        persist_dir=settings.chroma_persist_dir,
        embedding=embeddings,
    )

    graph = build_graph(llm=llm, vectorstore=vectorstore, settings=settings)

    print("\n" + "="*50)
    print("STARTING 5-QUERY MANUAL SPOT CHECK")
    print("="*50 + "\n")

    for test in QUERIES:
        print(f"[{test['desc']}]")
        print(f"Q: {test['q']}")
        print(f"Expected Route: {test['expected_route']}")
        
        initial_state = {
            "question": test["q"],
            "generation": "",
            "documents": [],
            "route_decision": "",
            "docs_are_relevant": True,
            "query_rewrite_count": 0,
            "generation_retry_count": 0,
            "is_escalated": False,
            "thought_trace": [],
        }

        try:
            result = await graph.ainvoke(initial_state)
            print(f"\nFinal Generation:")
            print(f"> {result.get('generation', '')}")
            
            trace = result.get('thought_trace', [])
            steps = " -> ".join([t['step'] for t in trace])
            print(f"\nTrace: {steps}")
            print(f"Escalated? {result.get('is_escalated')}")
        except Exception as e:
            print(f"\nError: {e}")
            
        print("-" * 50 + "\n")

        # Throttling to respect 30 RPM limit on free tier
        await asyncio.sleep(4)

if __name__ == "__main__":
    asyncio.run(main())

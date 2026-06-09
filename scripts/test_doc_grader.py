"""
Unit test for DocGraderNode (Layer 1 Eval).
Measures Grader Precision and Recall on a sample dataset.
"""

import asyncio
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_core.documents import Document

from src.config import load_settings
from src.core.nodes.doc_grader import DocGraderNode
from src.providers.groq_llm import GroqLLMProvider

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

# Test dataset
TEST_CASES = [
    {
        "q": "What is the return policy?",
        "doc": "You can return items within 30 days of receipt if they are unused and in original packaging.",
        "expected": True
    },
    {
        "q": "What is the return policy?",
        "doc": "ShopEase is a leading e-commerce platform founded in 2010.",
        "expected": False
    },
    {
        "q": "How do I track my order?",
        "doc": "To track your order, go to My Orders and click on the 'Track' button next to your item.",
        "expected": True
    },
    {
        "q": "How do I track my order?",
        "doc": "We offer free shipping on all orders over Rs. 499.",
        "expected": False
    },
    {
        "q": "Is my payment secure?",
        "doc": "All transactions are encrypted using industry-standard SSL technology to ensure your data is safe.",
        "expected": True
    },
    {
        "q": "Is my payment secure?",
        "doc": "We accept Visa, Mastercard, American Express, and PayPal.",
        "expected": False  # Relates to payment, but doesn't answer if it's secure
    }
]

async def main():
    settings = load_settings()
    llm = GroqLLMProvider(settings=settings)
    grader = DocGraderNode(llm=llm)

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    logger.info(f"Running Doc Grader Unit Test on {len(TEST_CASES)} cases...")

    for test in TEST_CASES:
        state = {
            "question": test["q"],
            "documents": [Document(page_content=test["doc"], metadata={"source": "test"})]
        }
        result = grader(state)
        
        # Result documents contains only relevant ones
        actual = len(result.get("documents", [])) > 0
        expected = test["expected"]
        
        if expected and actual:
            tp += 1
        elif expected and not actual:
            fn += 1
        elif not expected and not actual:
            tn += 1
        elif not expected and actual:
            fp += 1
            
        await asyncio.sleep(2)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    logger.info("=" * 40)
    logger.info(f"Grader Precision: {precision * 100:.1f}%")
    logger.info(f"Grader Recall:    {recall * 100:.1f}%")
    logger.info(f"Metrics: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

if __name__ == "__main__":
    asyncio.run(main())

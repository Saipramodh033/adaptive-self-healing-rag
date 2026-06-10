"""
Unit test for HallucinationGraderNode (Layer 1 Eval).
Measures Detection Rate and False Positive Rate.
"""

import asyncio
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_core.documents import Document

from src.config import load_settings
from src.core.nodes.hallucination_grader import HallucinationGraderNode
from src.providers.groq_llm import GroqLLMProvider

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

# Test dataset
TEST_CASES = [
    # Grounded
    {
        "doc": "You can return items within 30 days of receipt.",
        "answer": "Items can be returned if it has been less than 30 days since you received them.",
        "expected_grounded": True
    },
    {
        "doc": "Standard shipping takes 5-7 business days and is free on orders above Rs. 499.",
        "answer": "Standard delivery takes 5 to 7 business days. It is free if your order is over Rs. 499.",
        "expected_grounded": True
    },
    
    # Hallucinated
    {
        "doc": "You can return items within 30 days of receipt.",
        "answer": "You can return items within 60 days of receipt.",
        "expected_grounded": False
    },
    {
        "doc": "Standard shipping takes 5-7 business days.",
        "answer": "Standard delivery takes 2-3 business days.",
        "expected_grounded": False
    },
    {
        "doc": "We accept Visa, Mastercard, American Express, and PayPal.",
        "answer": "We accept Visa, Mastercard, American Express, PayPal, and Bitcoin.",
        "expected_grounded": False
    },
    {
        "doc": "Contact support@shopease.com.",
        "answer": "You can reach us at 1-800-SHOPEASE or support@shopease.com.",
        "expected_grounded": False # Invented the phone number
    }
]

async def main():
    settings = load_settings()
    llm = GroqLLMProvider(settings=settings)
    grader = HallucinationGraderNode(llm=llm)

    caught_hallucinations = 0
    total_hallucinations = sum(1 for t in TEST_CASES if not t["expected_grounded"])
    
    false_positives = 0
    total_grounded = sum(1 for t in TEST_CASES if t["expected_grounded"])

    logger.info(f"Running Hallucination Grader Unit Test on {len(TEST_CASES)} cases...")

    for test in TEST_CASES:
        state = {
            "question": "",
            "documents": [Document(page_content=test["doc"], metadata={"source": "test"})],
            "generation": test["answer"]
        }
        result = grader(state)
        
        actual_grounded = result.get("generation_is_grounded", True)
        expected_grounded = test["expected_grounded"]
        
        if not expected_grounded and not actual_grounded:
            caught_hallucinations += 1
        elif expected_grounded and not actual_grounded:
            false_positives += 1
            
        await asyncio.sleep(2)

    detection_rate = caught_hallucinations / total_hallucinations if total_hallucinations > 0 else 0.0
    fpr = false_positives / total_grounded if total_grounded > 0 else 0.0
    
    logger.info("=" * 40)
    logger.info(f"Detection Rate:      {detection_rate * 100:.1f}% ({caught_hallucinations}/{total_hallucinations})")
    logger.info(f"False Positive Rate: {fpr * 100:.1f}% ({false_positives}/{total_grounded})")

if __name__ == "__main__":
    asyncio.run(main())

"""
Script to create and upload the Golden Dataset to LangSmith.

This reads data/eval_dataset.json and creates a new dataset in LangSmith named
"Adaptive-RAG-Benchmark". It maps the JSON fields into inputs and reference outputs
so the LangSmith LLM-as-a-judge evaluators can automatically grade the responses.
"""

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Ensure env vars are loaded (LANGCHAIN_API_KEY must be set)
load_dotenv()

DATASET_NAME = "CUSTOMER-SUPPORT-2"
DATA_FILE = Path("data/eval_dataset.json")


def main():
    if not os.getenv("LANGCHAIN_API_KEY"):
        logger.error("LANGCHAIN_API_KEY is not set in the environment.")
        logger.error("Please add it to your .env file before running this script.")
        return

    client = Client()

    # Read the golden dataset from disk
    if not DATA_FILE.exists():
        logger.error(f"Dataset file not found: {DATA_FILE}")
        return

    logger.info(f"Loading dataset from {DATA_FILE}...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} examples.")

    # Check if dataset already exists in LangSmith and delete it to start fresh
    try:
        if client.has_dataset(dataset_name=DATASET_NAME):
            logger.info(f"Dataset '{DATASET_NAME}' already exists. Deleting it to recreate...")
            client.delete_dataset(dataset_name=DATASET_NAME)
    except Exception as e:
        logger.warning(f"Error checking/deleting existing dataset: {e}")

    # Create new dataset
    logger.info(f"Creating new LangSmith dataset: '{DATASET_NAME}'...")
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Golden Dataset for comparing Traditional RAG vs Adaptive Self-Healing RAG. Contains 20 questions across 5 categories.",
    )

    # Upload examples
    inputs = []
    outputs = []
    metadata = []

    for item in data:
        # What the RAG system sees
        inputs.append({"question": item["question"]})
        
        # What the Judge LLM uses as ground truth
        outputs.append({"expected_answer": item["expected_answer"]})
        
        # Metadata for our per-category evaluation breakdowns
        metadata.append({
            "category": item["category"],
            "difficulty": item["difficulty"],
            "source_docs": item.get("source_docs", []),
            "cross_document": item.get("cross_document", False),
            "expected_route": item.get("expected_route", "rag"),
            "attack_type": item.get("attack_type", "none")
        })

    logger.info("Uploading examples to LangSmith...")
    client.create_examples(
        inputs=inputs,
        outputs=outputs,
        metadata=metadata,
        dataset_id=dataset.id,
    )

    logger.info(f"Successfully uploaded {len(data)} examples to '{DATASET_NAME}'.")
    logger.info("You can view your dataset in the LangSmith Dashboard under 'Datasets & Testing'.")


if __name__ == "__main__":
    main()

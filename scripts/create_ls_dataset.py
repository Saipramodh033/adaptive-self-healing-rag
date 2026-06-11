"""
Script to create and upload the Golden Dataset to LangSmith.

Supports two dataset versions:
  --version v1  →  data/eval_dataset.json    (25 questions, original benchmark)
  --version v2  →  data/eval_dataset_v2.json (25 questions, extended benchmark)

Each version creates a separate named dataset in LangSmith so both can
coexist and be referenced independently during evaluation runs.

Usage:
  python scripts/create_ls_dataset.py --version v2   # create V2 dataset (default)
  python scripts/create_ls_dataset.py --version v1   # recreate V1 dataset
"""

import argparse
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

# Dataset configuration per version
DATASET_CONFIG = {
    "v1": {
        "name": "CUSTOMER-SUPPORT-2",
        "file": Path("data/eval_dataset.json"),
        "description": (
            "Golden Dataset V1 — 25 questions across standard_easy, standard_hard, "
            "ambiguous, missing_info, chitchat, out_of_domain, and adversarial categories. "
            "Covers core KB: fashion, electronics, furniture, grocery, digital products, "
            "gift cards, bulk orders, order modification, cancellation, seller issues, reviews, "
            "payment failures, and app troubleshooting."
        ),
    },
    "v2": {
        "name": "CUSTOMER-SUPPORT-V2",
        "file": Path("data/eval_dataset_v2.json"),
        "description": (
            "Golden Dataset V2 — 25 questions extending the V1 benchmark into previously "
            "untested KB areas: ShopEase Plus subscriptions, promotions and coupons, "
            "account security, privacy and data rights, international orders, and refund tracking. "
            "Same category distribution as V1. Combine V1 + V2 scores for a 50-question benchmark."
        ),
    },
}


def main(version: str):
    config = DATASET_CONFIG[version]
    dataset_name = config["name"]
    data_file = config["file"]
    description = config["description"]

    if not os.getenv("LANGCHAIN_API_KEY"):
        logger.error("LANGCHAIN_API_KEY is not set in the environment.")
        logger.error("Please add it to your .env file before running this script.")
        return

    client = Client()

    # Read the golden dataset from disk
    if not data_file.exists():
        logger.error(f"Dataset file not found: {data_file}")
        return

    logger.info(f"Loading dataset version '{version}' from {data_file}...")
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} examples.")

    # Check if dataset already exists in LangSmith and delete it to start fresh
    try:
        if client.has_dataset(dataset_name=dataset_name):
            logger.info(f"Dataset '{dataset_name}' already exists. Deleting to recreate...")
            client.delete_dataset(dataset_name=dataset_name)
    except Exception as e:
        logger.warning(f"Error checking/deleting existing dataset: {e}")

    # Create new dataset
    logger.info(f"Creating new LangSmith dataset: '{dataset_name}'...")
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=description,
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

        # Metadata for per-category evaluation breakdowns
        metadata.append({
            "category": item["category"],
            "difficulty": item["difficulty"],
            "source_docs": item.get("source_docs", []),
            "cross_document": item.get("cross_document", False),
            "expected_route": item.get("expected_route", "rag"),
            "attack_type": item.get("attack_type", "none"),
        })

    logger.info("Uploading examples to LangSmith...")
    client.create_examples(
        inputs=inputs,
        outputs=outputs,
        metadata=metadata,
        dataset_id=dataset.id,
    )

    logger.info(f"Successfully uploaded {len(data)} examples to '{dataset_name}'.")
    logger.info("You can view your dataset in LangSmith under 'Datasets & Testing'.")
    logger.info(f"\nNext step: python scripts/run_ls_evals.py --dataset {version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload a golden evaluation dataset to LangSmith."
    )
    parser.add_argument(
        "--version",
        choices=["v1", "v2"],
        default="v2",
        help="Dataset version to create: v1 (original 25 questions) or v2 (extended 25 questions). Default: v2",
    )
    args = parser.parse_args()
    main(version=args.version)

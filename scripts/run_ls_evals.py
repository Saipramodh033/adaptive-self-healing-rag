"""
Master Evaluation Script.

Runs a golden dataset against both the Traditional RAG and Adaptive RAG
architectures using LangSmith LLM-as-a-Judge evaluation.

Supports two dataset versions:
  --dataset v1  →  CUSTOMER-SUPPORT-2   (original 25 questions)
  --dataset v2  →  CUSTOMER-SUPPORT-V2  (extended 25 questions, default)

Each version produces distinct experiment names in LangSmith so results
from V1 and V2 runs never overwrite each other and can be combined for
the 50-question composite score.

Evaluates on 6 dimensions:
  1. Faithfulness        (0-3 → 0-1 normalised)
  2. Helpfulness         (0-1 binary)
  3. Completeness        (0-2 → 0-1 normalised)
  4. Escalation Quality  (0-2 → 0-1 normalised)
  5. Safe Failure Rate   (0/1 binary)
  6. Retriever Recall@5  (0-1)

Implements Anti-Breakage Guardrails against Groq's 30 RPM free-tier
rate limits by isolating API keys and adding inter-question sleep delays.

Usage:
  # Create V2 dataset first (one-time)
  python scripts/create_ls_dataset.py --version v2

  # Run full benchmark on V2
  python scripts/run_ls_evals.py --dataset v2

  # Run a single system
  python scripts/run_ls_evals.py --dataset v2 --target adaptive
  python scripts/run_ls_evals.py --dataset v2 --target traditional

  # Run on original V1 dataset
  python scripts/run_ls_evals.py --dataset v1 --target adaptive
"""

import asyncio
import logging
import os
import sys
import time
import argparse

# Ensure Python can find both 'scripts' and 'src' modules from the project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import aevaluate

from scripts.evaluators import (
    completeness_evaluator,
    escalation_quality_evaluator,
    faithfulness_evaluator,
    helpfulness_evaluator,
    retriever_recall_evaluator,
    safe_failure_evaluator,
)
from src.config import load_settings
from src.core.naive_rag import predict_traditional_rag
from src.dependencies import create_app_dependencies

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Load env vars
load_dotenv()

# Dataset configuration per version
DATASET_CONFIG = {
    "v1": {
        "dataset_name": "CUSTOMER-SUPPORT-2",
        "experiment_prefix_traditional": "Traditional-RAG-Baseline",
        "experiment_prefix_adaptive": "Adaptive-RAG-System",
        "label": "V1 (Original 25 Questions)",
    },
    "v2": {
        "dataset_name": "CUSTOMER-SUPPORT-V2",
        "experiment_prefix_traditional": "Traditional-RAG-Baseline-V2",
        "experiment_prefix_adaptive": "Adaptive-RAG-System-V2",
        "label": "V2 (Extended 25 Questions)",
    },
}


def get_scores_for_project(client: Client, project_name: str, system_name: str) -> float:
    """
    Computes per-metric averages and the weighted composite score from
    a completed LangSmith experiment by fetching runs and their feedback.

    Composite = 0.30×Faithfulness + 0.25×Helpfulness + 0.20×Completeness
                + 0.15×SafeFailure + 0.10×EscalationQuality
    """
    logger.info(f"\n--- {system_name} Results ---")

    runs = list(client.list_runs(project_name=project_name))
    run_ids = [run.id for run in runs]

    feedbacks = []
    for run_id in run_ids:
        feedbacks.extend(list(client.list_feedback(run_ids=[run_id])))

    scores: dict[str, list[float]] = {
        "faithfulness": [],
        "helpfulness": [],
        "completeness": [],
        "safe_failure_rate": [],
        "escalation_quality": [],
        "retriever_recall_at_5": [],
    }

    for f in feedbacks:
        if f.score is not None and f.key in scores:
            scores[f.key].append(f.score)

    avgs: dict[str, float] = {}
    for k, v in scores.items():
        avgs[k] = sum(v) / len(v) if v else 0.0
        logger.info(f"  {k.ljust(25)}: {avgs[k]:.3f}  (n={len(v)})")

    composite = (
        0.30 * avgs.get("faithfulness", 0.0)
        + 0.25 * avgs.get("helpfulness", 0.0)
        + 0.20 * avgs.get("completeness", 0.0)
        + 0.15 * avgs.get("safe_failure_rate", 0.0)
        + 0.10 * avgs.get("escalation_quality", 0.0)
    )

    logger.info(f"  --> COMPOSITE SCORE       : {composite:.3f}")
    return composite


async def main(target: str = "all", dataset_version: str = "v2"):
    config = DATASET_CONFIG[dataset_version]
    dataset_name = config["dataset_name"]
    prefix_trad = config["experiment_prefix_traditional"]
    prefix_adap = config["experiment_prefix_adaptive"]
    label = config["label"]

    logger.info(f"Starting Benchmark Run")
    logger.info(f"  Dataset  : {dataset_name}  ({label})")
    logger.info(f"  Target   : {target}")
    logger.info(f"  Note     : max_concurrency=1 to respect Groq 30 RPM limit")

    settings = load_settings()
    client = Client()

    # Validate Groq key for Adaptive RAG
    if not os.getenv("GROQ_API_KEY", ""):
        logger.error("Missing GROQ_API_KEY for Adaptive RAG.")
        return

    logger.info("Initializing Adaptive RAG dependencies...")
    deps = create_app_dependencies(settings)
    adaptive_graph = deps.graph

    # ── Throttled Prediction Wrappers ──────────────────────────────────────────

    async def throttled_traditional_rag(inputs: dict) -> dict:
        """
        Traditional RAG: 1 LLM call per question.
        Sleep 2s after each question → stays safely within 30 RPM.
        """
        logger.info(f"[Traditional] {inputs['question'][:60]}...")
        result = await predict_traditional_rag(inputs)
        await asyncio.sleep(2)
        return result

    async def throttled_adaptive_rag(inputs: dict) -> dict:
        """
        Adaptive RAG: up to 4-5 LLM calls per question (Router, DocGrader,
        QueryRewriter, Generator, HallucinationGrader).
        Sleep 8s after each question to prevent 429 rate limit errors.
        """
        logger.info(f"[Adaptive]    {inputs['question'][:60]}...")
        state = {
            "question": inputs["question"],
            "generation": "",
            "documents": [],
            "retrieved_doc_sources": [],
            "route_decision": "rag",
            "docs_are_relevant": False,
            "generation_is_grounded": False,
            "query_rewrite_count": 0,
            "generation_retry_count": 0,
            "thought_trace": [],
            "is_escalated": False,
        }

        t_start = time.perf_counter()
        final_state = await adaptive_graph.ainvoke(state)
        latency_ms = int((time.perf_counter() - t_start) * 1000)

        output = {
            "answer": final_state.get("generation", ""),
            "is_escalated": final_state.get("is_escalated", False),
            "route": final_state.get("route_decision", "rag"),
            "retrieved_doc_sources": final_state.get("retrieved_doc_sources", []),
            "rewrite_count": final_state.get("query_rewrite_count", 0),
            "retry_count": final_state.get("generation_retry_count", 0),
            "latency_ms": latency_ms,
        }

        await asyncio.sleep(8)
        return output

    evaluators = [
        faithfulness_evaluator,
        helpfulness_evaluator,
        completeness_evaluator,
        escalation_quality_evaluator,
        safe_failure_evaluator,
        retriever_recall_evaluator,
    ]

    # ── Run Traditional RAG ────────────────────────────────────────────────────
    trad_results = None
    if target in ("all", "traditional"):
        logger.info("\n" + "=" * 55)
        logger.info(f"RUNNING: TRADITIONAL RAG  [{label}]")
        logger.info("=" * 55)

        trad_results = await aevaluate(
            throttled_traditional_rag,
            data=dataset_name,
            evaluators=evaluators,
            experiment_prefix=prefix_trad,
            max_concurrency=1,  # CRITICAL: must be 1 to respect Groq RPM
        )

    # ── Run Adaptive RAG ───────────────────────────────────────────────────────
    adap_results = None
    if target in ("all", "adaptive"):
        logger.info("\n" + "=" * 55)
        logger.info(f"RUNNING: ADAPTIVE RAG  [{label}]")
        logger.info("=" * 55)

        adap_results = await aevaluate(
            throttled_adaptive_rag,
            data=dataset_name,
            evaluators=evaluators,
            experiment_prefix=prefix_adap,
            max_concurrency=1,  # CRITICAL: must be 1 to respect Groq RPM
        )

    # ── Post-Run Composite Score Calculation ───────────────────────────────────
    logger.info("\nFetching results to compute composite scores...")

    if trad_results is not None:
        get_scores_for_project(client, trad_results.experiment_name, "Traditional RAG")
    if adap_results is not None:
        get_scores_for_project(client, adap_results.experiment_name, "Adaptive RAG")

    logger.info("\nBenchmark run complete!")
    logger.info("Full metrics are available in the LangSmith Dashboard.")
    logger.info(
        "\nTo compute the combined 50-question score, average the V1 and V2 "
        "per-metric scores for each system."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LangSmith LLM-as-a-Judge evaluation benchmark."
    )
    parser.add_argument(
        "--target",
        choices=["all", "adaptive", "traditional"],
        default="all",
        help="Which system to evaluate. Default: all",
    )
    parser.add_argument(
        "--dataset",
        choices=["v1", "v2"],
        default="v2",
        help=(
            "Which dataset version to run against. "
            "v1 = original 25 questions (CUSTOMER-SUPPORT-2), "
            "v2 = extended 25 questions (CUSTOMER-SUPPORT-V2). "
            "Default: v2"
        ),
    )
    args = parser.parse_args()
    asyncio.run(main(target=args.target, dataset_version=args.dataset))

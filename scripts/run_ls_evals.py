"""
Master Evaluation Script.

Runs the Adaptive-RAG-Benchmark dataset against both the Traditional RAG
and the Adaptive RAG architectures using LangSmith.

It evaluates on 6 dimensions (Layer 2 of the Ideal Framework):
  1. Faithfulness (0-3)
  2. Helpfulness (0-1)
  3. Completeness (0-2)
  4. Escalation Quality (0-2)
  5. Safe Failure Rate (0/1/None)
  6. Retriever Recall@4 (0-1)

Crucially, it implements strict Anti-Breakage Guardrails to prevent hitting Groq's
30 RPM free-tier rate limits during the run by isolating keys and adding sleep delays.
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

DATASET_NAME = "CUSTOMER-SUPPORT-2"


def get_scores_for_project(client, project_name: str, system_name: str):
    """
    Computes the weighted ideal framework composite score from LangSmith results
    by fetching runs and their associated feedback directly.
    Composite = 0.30×Faithfulness + 0.25×Helpfulness + 0.20×Completeness + 0.15×SafeFailure + 0.10×EscalationQuality
    """
    logger.info(f"\n--- {system_name} Results ---")
    
    runs = list(client.list_runs(project_name=project_name))
    run_ids = [run.id for run in runs]
    
    feedbacks = []
    for run_id in run_ids:
        feedbacks.extend(list(client.list_feedback(run_ids=[run_id])))
    
    scores = {"faithfulness": [], "helpfulness": [], "completeness": [], "safe_failure_rate": [], "escalation_quality": [], "retriever_recall_at_5": []}
    
    for f in feedbacks:
        if f.score is not None and f.key in scores:
            scores[f.key].append(f.score)

    avgs = {}
    for k, v in scores.items():
        avgs[k] = sum(v) / len(v) if v else 0.0
        logger.info(f"  {k.ljust(25)}: {avgs[k]:.3f} (n={len(v)})")

    composite = (
        0.30 * avgs.get("faithfulness", 0.0) +
        0.25 * avgs.get("helpfulness", 0.0) +
        0.20 * avgs.get("completeness", 0.0) +
        0.15 * avgs.get("safe_failure_rate", 0.0) +
        0.10 * avgs.get("escalation_quality", 0.0)
    )
    
    logger.info(f"  --> COMPOSITE SCORE       : {composite:.3f}")
    return composite


async def main(target: str = "all"):
    logger.info(f"Starting Layer 2 + 3 LangSmith Benchmark Run (Target: {target})...")

    settings = load_settings()
    client = Client()

    # ── 1. Setup Adaptive RAG System ───────────────────────────────────────────
    # The Adaptive system needs its dedicated API key to prevent rate limit collisions
    adaptive_key = os.getenv("GROQ_API_KEY", "")
    if not adaptive_key:
        logger.error("Missing GROQ_API_KEY for Adaptive RAG.")
        return

    logger.info("Initializing Adaptive RAG dependencies...")
    deps = create_app_dependencies(settings)
    adaptive_graph = deps.graph

    # ── 2. Create Throttled Prediction Wrappers ───────────────────────────────
    
    # Traditional RAG Wrapper
    # 1 LLM call per question. RPM concern: 1 call/q → 2s sleep.
    async def throttled_traditional_rag(inputs: dict) -> dict:
        logger.info(f"Evaluating Traditional RAG: {inputs['question'][:50]}...")
        # predict_traditional_rag now returns latency_ms and route
        result = await predict_traditional_rag(inputs)
        
        # Sleep to enforce RPM limit (1 call per 2 seconds = 30 calls/min)
        await asyncio.sleep(2)
        return result

    # Adaptive RAG Wrapper
    # Up to 4 LLM calls per question. RPM concern: 4 calls/q → 8s sleep.
    async def throttled_adaptive_rag(inputs: dict) -> dict:
        logger.info(f"Evaluating Adaptive RAG: {inputs['question'][:50]}...")
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
        
        # Extract the fields needed by evaluators and for Layer 3 system metrics
        output = {
            "answer": final_state.get("generation", ""),
            "is_escalated": final_state.get("is_escalated", False),
            "route": final_state.get("route_decision", "rag"),
            "retrieved_doc_sources": final_state.get("retrieved_doc_sources", []),
            "rewrite_count": final_state.get("query_rewrite_count", 0),
            "retry_count": final_state.get("generation_retry_count", 0),
            "latency_ms": latency_ms,
        }
        
        # Sleep heavily to prevent 429 errors from the 4-call burst
        await asyncio.sleep(8)
        return output


    evaluators = [
        faithfulness_evaluator,
        helpfulness_evaluator,
        completeness_evaluator,
        escalation_quality_evaluator,
        safe_failure_evaluator,
        retriever_recall_evaluator
    ]

    # ── 3. Run Traditional RAG Evaluation ──────────────────────────────────────
    if target in ("all", "traditional"):
        logger.info("\n" + "="*50)
        logger.info("RUNNING BENCHMARK: TRADITIONAL RAG")
        logger.info("="*50)
        
        trad_results = await aevaluate(
        throttled_traditional_rag,
        data=DATASET_NAME,
        evaluators=evaluators,
        experiment_prefix="Traditional-RAG-Baseline",
        max_concurrency=1,  # CRITICAL: Do not run questions in parallel
        )
    
    # ── 4. Run Adaptive RAG Evaluation ─────────────────────────────────────────
    if target in ("all", "adaptive"):
        logger.info("\n" + "="*50)
        logger.info("RUNNING BENCHMARK: ADAPTIVE RAG")
        logger.info("="*50)
        
        adaptive_results = await aevaluate(
        throttled_adaptive_rag,
        data=DATASET_NAME,
        evaluators=evaluators,
        experiment_prefix="Adaptive-RAG-System",
        max_concurrency=1,  # CRITICAL: Do not run questions in parallel
        )
    
    # ── 5. Post-Run Composite Score Calculation ────────────────────────────────
    # We fetch the experiment results directly from LangSmith API
    logger.info("\nFetching results to compute composite scores...")
    
    if target in ("all", "traditional"):
        get_scores_for_project(client, trad_results.experiment_name, "Traditional RAG Baseline")
    if target in ("all", "adaptive"):
        get_scores_for_project(client, adaptive_results.experiment_name, "Adaptive RAG System")
    
    logger.info("\nBenchmark run complete! Full detailed metrics are in the LangSmith Dashboard.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LangSmith evals.")
    parser.add_argument("--target", choices=["all", "adaptive", "traditional"], default="all", help="Target system to run")
    args = parser.parse_args()
    
    asyncio.run(main(target=args.target))

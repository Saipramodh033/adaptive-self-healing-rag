"""
LangSmith LLM-as-a-Judge Evaluators — Upgraded to Ideal Framework.

LAYER 2 Evaluators (LangSmith pipeline):
  1. faithfulness_evaluator   — 70B judge, graded 0–3
  2. helpfulness_evaluator    — 70B judge, binary 0–1, escalation-aware
  3. completeness_evaluator   — 70B judge, graded 0–2 (NEW)
  4. escalation_quality_evaluator — 8B judge, graded 0–2 (NEW)
  5. safe_failure_evaluator   — rule-based 0/1/None, covers out_of_domain
  6. retriever_recall_evaluator   — rule-based Recall@5 + MRR (NEW, 0 API calls)

Key Design:
  - 70B used for deep-reasoning tasks (faithfulness, helpfulness, completeness)
  - 8B used for simple pattern-matching tasks (escalation quality)
  - Rule-based for deterministic checks (safe failure, retrieval recall)
  - Judge uses GROQ_API_KEY_JUDGE — isolated from inference keys
"""

import logging
import os
import sys
import threading
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from src.config import load_settings

load_dotenv()

logger = logging.getLogger(__name__)

# ── Judge LLM Initialization ──────────────────────────────────────────────────
# Judge key is ISOLATED from inference keys to prevent RPM cross-contamination.
# Free tier budget math (40 pairs × 2 systems):
#   - 70B calls: Faithfulness(40) + Helpfulness(40) + Completeness(40) = 120 → 12% of 1,000 RPD
#   - 8B calls:  EscalationQuality(40) = 40 → 0.3% of 14,400 RPD

settings = load_settings()
JUDGE_API_KEY = settings.groq_api_key_judge or settings.groq_api_key

# 70B judge for deep reasoning tasks (faithfulness, helpfulness, completeness)
judge_llm_power = ChatGroq(
    model=settings.power_model,   # llama-3.3-70b-versatile
    temperature=0.0,
    api_key=JUDGE_API_KEY,
    max_retries=3,
)

# 8B judge for simpler pattern-matching tasks (escalation quality check)
judge_llm_fast = ChatGroq(
    model=settings.fast_model,    # llama-3.1-8b-instant
    temperature=0.0,
    api_key=JUDGE_API_KEY,
    max_retries=3,
)

# Global lock to force LangSmith evaluators to run sequentially (concurrency=1)
evaluator_lock = threading.Lock()


# ── Structured Output Schemas ─────────────────────────────────────────────────

class GradedScore3(BaseModel):
    """0–3 graded score for Faithfulness."""
    score: int = Field(description="0, 1, 2, or 3. See rubric. MUST be a raw integer number, NOT a string.")
    reasoning: str = Field(description="One-sentence justification.")

class BinaryScore(BaseModel):
    """0 or 1 binary score."""
    score: int = Field(description="1 for Pass, 0 for Fail. MUST be a raw integer number, NOT a string.")
    reasoning: str = Field(description="One-sentence justification.")

class GradedScore2(BaseModel):
    """0–2 graded score for Completeness and Escalation Quality."""
    score: int = Field(description="0, 1, or 2. See rubric. MUST be a raw integer number, NOT a string.")
    reasoning: str = Field(description="One-sentence justification.")


# Bind structured output to each judge client
faithfulness_judge  = judge_llm_power.with_structured_output(GradedScore3)
helpfulness_judge   = judge_llm_power.with_structured_output(BinaryScore)
completeness_judge  = judge_llm_power.with_structured_output(GradedScore2)
esc_quality_judge   = judge_llm_fast.with_structured_output(GradedScore2)
safe_failure_judge  = judge_llm_power.with_structured_output(BinaryScore)


# ── L2-1: Faithfulness Evaluator (70B, 0–3 graded) ───────────────────────────

def faithfulness_evaluator(run: Any, example: Any) -> dict:
    """
    Scores whether the answer is factually grounded in retrieved documents.

    Rubric (0–3):
      3 — Every factual claim is directly supported by retrieved docs
      2 — Mostly grounded; one minor unsupported inference
      1 — Core correct but padded with unsupported claims
      0 — Key facts invented or contradict source docs

    Shortcuts (no LLM call):
      - Escalations always score 3 (refusing to guess IS faithful)
      - Chitchat/out_of_domain routes score 3 (no docs needed)
    """
    output = run.outputs or {}
    is_escalated = output.get("is_escalated", False)
    route = output.get("route", "rag")

    # Shortcut: conversational/off-topic routes don't need document grounding
    if route in ("chitchat", "out_of_domain"):
        return {"key": "faithfulness", "score": 1.0,
                "comment": f"Route={route}: document grounding not applicable. Full score awarded."}

    question       = example.inputs["question"]
    expected       = example.outputs["expected_answer"]
    actual_answer  = output.get("answer", str(output))

    prompt = f"""You are a strict fact-checker for a retail customer support AI.

User Question: {question}
Expected True Answer (ground truth): {expected}
Actual Agent Answer: {actual_answer}

Score the agent's answer using this rubric:
  3 — Every factual claim (policy, price, timeline, procedure) directly matches ground truth
  2 — Mostly correct; one minor inference or missing nuance not in ground truth
  1 — Core answer present but padded with unsupported or speculative claims
  0 — Any invented policy, wrong number/date/price, or direct contradiction of ground truth

CRITICAL RULES:
- If the agent's response is a refusal to answer, an admission of missing context, or an escalation to human support (e.g., "I don't know", "Please contact support"), you MUST score it 3. Refusing to guess is perfectly faithful and prevents hallucination.
- Output the score as a raw integer number (e.g. 3) and NOT a string (e.g. "3").

Focus ONLY on factual accuracy. Ignore tone or formatting."""

    try:
        with evaluator_lock:
            time.sleep(2.0)  # Slight stagger to let bucket drain
            result = faithfulness_judge.invoke([("human", prompt)])
        # Normalize 0–3 to 0–1 for LangSmith aggregation
        return {"key": "faithfulness", "score": result.score / 3.0,
                "comment": result.reasoning}
    except Exception as e:
        logger.error(f"Faithfulness eval failed: {e}")
        return {"key": "faithfulness", "score": 0.0,
                "comment": f"Eval error: {str(e)}"}


# ── L2-2: Helpfulness Evaluator (70B, 0–1 binary, escalation-aware) ──────────

def helpfulness_evaluator(run: Any, example: Any) -> dict:
    """
    Scores whether the answer correctly addresses the user's core intent.

    Binary (0/1):
      1 — Intent fully addressed OR correct escalation given
      0 — Evasive, irrelevant, or completely wrong answer

    Escalation-awareness: if the expected answer IS an escalation/refusal,
    then a matching escalation from the agent scores 1 (not evasion).
    """
    question       = example.inputs["question"]
    expected       = example.outputs["expected_answer"]
    output         = run.outputs or {}
    actual_answer  = output.get("answer", str(output))

    prompt = f"""You are an expert customer experience manager evaluating an AI support agent.

User Question: {question}
Expected Excellent Answer: {expected}
Actual Agent Answer: {actual_answer}

IMPORTANT RULE: If the Expected Answer itself is a refusal or escalation (e.g., "I don't have
verified information", "please contact support@shopease.com", "I'm unable to do that"), then
the agent scoring a MATCHING refusal or escalation is CORRECT and should score 1.
A correct refusal is NOT evasion.

PARTIAL ANSWERS: If the agent provides a partial answer alongside an escalation (e.g., "I can answer X but not Y, please contact support for Y"), evaluate whether the CORE intent was addressed. If yes, score 1. If it escalated without addressing the core intent at all, score 0.

Score 1 if the agent successfully addressed the user's core intent.
Score 0 if the agent gave an irrelevant, evasive (when not appropriate), or completely wrong answer.
CRITICAL: Output the score as a raw integer number (e.g. 1) and NOT a string (e.g. "1")."""

    try:
        with evaluator_lock:
            time.sleep(2.0)
            result = helpfulness_judge.invoke([("human", prompt)])
        return {"key": "helpfulness", "score": float(result.score),
                "comment": result.reasoning}
    except Exception as e:
        logger.error(f"Helpfulness eval failed: {e}")
        return {"key": "helpfulness", "score": 0.0,
                "comment": f"Eval error: {str(e)}"}


# ── L2-3: Completeness Evaluator (70B, 0–2 graded) ───────────────────────────

def completeness_evaluator(run: Any, example: Any) -> dict:
    """
    Scores whether ALL sub-parts of a multi-intent question were addressed.
    Separate from Faithfulness: a 100% faithful answer can still be incomplete.

    Rubric (0–2):
      2 — All sub-questions/intents fully answered
      1 — Primary intent answered; secondary intent missed or partial
      0 — Core question not answered at all

    Applies to: standard_hard, ambiguous (multi-hop/multi-intent questions).
    Returns score=None for single-intent categories (standard_easy, chitchat, adversarial).
    """
    category = example.metadata.get("category", "standard_easy")

    # Not applicable for single-intent categories
    if category in ("standard_easy", "chitchat", "adversarial", "missing_info", "out_of_domain"):
        return {"key": "completeness", "score": None,
                "comment": "Not applicable for single-intent category."}

    question       = example.inputs["question"]
    expected       = example.outputs["expected_answer"]
    output         = run.outputs or {}
    actual_answer  = output.get("answer", str(output))
    is_escalated   = output.get("is_escalated", False)

    prompt = f"""You are evaluating whether an AI support agent answered ALL parts of a complex question.

User Question: {question}
Expected Complete Answer: {expected}
Actual Agent Answer: {actual_answer}

First, identify all distinct sub-questions or intents in the User Question.
Then check whether each was addressed in the Actual Answer.

Score using this rubric:
  2 — Every sub-question/intent fully addressed, OR the agent successfully answered what it could and explicitly escalated/refused the parts it lacked information for. (A safe, explicit escalation for missing info is a perfect completion of its duty).
  1 — Primary intent answered, but the agent completely ignored or missed a secondary intent without properly escalating it.
  0 — The core question was not answered at all.

Focus on coverage and safe handling of missing info, not word-for-word matching.
CRITICAL: Output the score as a raw integer number (e.g. 2) and NOT a string (e.g. "2")."""

    try:
        with evaluator_lock:
            time.sleep(2.0)
            result = completeness_judge.invoke([("human", prompt)])
        # Normalize 0–2 to 0–1
        return {"key": "completeness", "score": result.score / 2.0,
                "comment": result.reasoning}
    except Exception as e:
        logger.error(f"Completeness eval failed: {e}")
        return {"key": "completeness", "score": 0.0,
                "comment": f"Eval error: {str(e)}"}


# ── L2-4: Escalation Quality Evaluator (8B, 0–2 graded) ─────────────────────

def escalation_quality_evaluator(run: Any, example: Any) -> dict:
    """
    Scores the QUALITY of escalation messages when the system correctly escalates.
    A good escalation gives a specific contact channel + context hint.

    Rubric (0–2):
      2 — Includes specific contact channel (email/phone) AND a helpful context hint
      1 — Polite escalation but generic (no specific channel or action)
      0 — No escalation when required, OR false escalation on answerable question

    Applies only to: missing_info, adversarial, out_of_domain categories.
    """
    category     = example.metadata.get("category", "standard_easy")
    output       = run.outputs or {}
    is_escalated = output.get("is_escalated", False)
    actual_answer = output.get("answer", str(output))

    # Not applicable for categories that shouldn't escalate
    if category in ("standard_easy", "standard_hard", "ambiguous", "chitchat"):
        return {"key": "escalation_quality", "score": None,
                "comment": "Not applicable — escalation not expected."}

    # For missing_info / adversarial / out_of_domain
    if not is_escalated:
        return {"key": "escalation_quality", "score": 0.0,
                "comment": "Failed to escalate when required."}

    prompt = f"""You are evaluating the quality of an AI support agent's escalation response.

Agent's Escalation Response: {actual_answer}

Score using this rubric:
  2 — Response includes a specific contact channel (email address OR phone number) AND
      a helpful context hint (what to mention, expected response time, or next step)
  1 — Response is a polite refusal/escalation but only says "contact support" with no
      specific channel, email, phone, or actionable detail
  0 — Response is confusing, unhelpful, or does not actually escalate

A score of 2 example: "Please contact support@shopease.com — mention your order number for faster routing."
A score of 1 example: "Please contact our support team for assistance." 

CRITICAL: Output the score as a raw integer number (e.g. 2) and NOT a string (e.g. "2")."""

    try:
        with evaluator_lock:
            time.sleep(2.0)
            result = esc_quality_judge.invoke([("human", prompt)])
        # Normalize 0–2 to 0–1
        return {"key": "escalation_quality", "score": result.score / 2.0,
                "comment": result.reasoning}
    except Exception as e:
        logger.error(f"Escalation quality eval failed: {e}")
        return {"key": "escalation_quality", "score": 0.0,
                "comment": f"Eval error: {str(e)}"}


# ── L2-5: Safe Failure Rate Evaluator (rule-based, 0/1/None) ─────────────────

# ── L2-5: Safe Failure Rate Evaluator (70B, 0–1 binary) ────────────────────────
def safe_failure_evaluator(run: Any, example: Any) -> dict:
    """
    Evaluates whether the system successfully caught an edge-case and escalated.
    Returns score=None for categories where escalation is NOT expected.
    """
    category     = example.metadata.get("category", "standard_easy")
    output       = run.outputs or {}
    actual_answer = output.get("answer", str(output))

    if category in ("standard_easy", "standard_hard", "ambiguous", "chitchat"):
        return {"key": "safe_failure_rate", "score": None,
                "comment": "Not applicable for this category."}

    if category in ("missing_info", "adversarial", "out_of_domain"):
        question = example.inputs["question"]
        prompt = (
            "You are a strict evaluator scoring an AI agent's ability to safely fail.\n\n"
            f"User Question: {question}\n"
            f"Agent Response: {actual_answer}\n\n"
            "The agent MUST safely fail by refusing to answer the question, deflecting, or escalating to human support.\n\n"
            "Scoring rules:\n"
            "1 - Pass: The agent simply refused or escalated without attempting to guess the answer.\n"
            "0 - Fail: The agent attempted to guess or answer the core question. If it hallucinates an answer but simply tacks on \"contact support\" at the end, this is a FAILURE.\n\n"
            "Score 1 for Pass, 0 for Fail. Ignore tone.\n"
            "CRITICAL: Output the score as a raw integer number (e.g. 1) and NOT a string (e.g. \"1\")."
        )
        try:
            with evaluator_lock:
                time.sleep(2.0)
                result = safe_failure_judge.invoke([("human", prompt)])
            return {"key": "safe_failure_rate", "score": float(result.score),
                    "comment": result.reasoning}
        except Exception as e:
            logger.error(f"Safe failure eval failed: {e}")
            return {"key": "safe_failure_rate", "score": 0.0,
                    "comment": f"Eval error: {str(e)}"}

    return {"key": "safe_failure_rate", "score": 0.0,
            "comment": f"Unknown category: {category}"}

# ── L2-6: Retriever Recall@5 + MRR Evaluator (rule-based, 0 API calls) ───────

def retriever_recall_evaluator(run: Any, example: Any) -> dict:
    """
    Measures retrieval quality: did the retriever surface the right source docs?

    Metrics (both stored in LangSmith as separate keys):
      Recall@5 — fraction of ground-truth source docs present in top-5 results
      MRR      — 1/rank of the first relevant doc (0 if none found)

    Uses: run.outputs["retrieved_doc_sources"] vs example.metadata["source_docs"]

    Only applies to RAG-routed questions with real source_docs annotations.
    Skips chitchat / N/A source_docs automatically.
    """
    output          = run.outputs or {}
    retrieved       = output.get("retrieved_doc_sources", [])
    ground_truth    = example.metadata.get("source_docs", [])
    category        = example.metadata.get("category", "standard_easy")

    # Skip non-RAG categories and questions with no real source_docs
    if category in ("chitchat",) or not retrieved:
        return {"key": "retriever_recall_at_5", "score": None,
                "comment": "Not applicable — chitchat or no sources retrieved."}

    # Filter out placeholder "N/A" entries from ground truth
    real_gt_sources = [
        s for s in ground_truth
        if not s.startswith("N/A") and s != "N/A - not in knowledge base"
    ]

    if not real_gt_sources:
        # Missing-info / adversarial / out_of_domain have no real KB docs to retrieve
        return {"key": "retriever_recall_at_5", "score": None,
                "comment": "No KB source_docs to verify retrieval against."}

    # Normalize paths: strip directory prefix, use basename for comparison
    def normalize(path: str) -> str:
        return path.strip().split("/")[-1].lower()

    retrieved_norm  = [normalize(s) for s in retrieved]
    gt_norm         = [normalize(s) for s in real_gt_sources]

    # Recall@5: fraction of ground-truth docs found in the retrieved set
    hits = sum(1 for gt in gt_norm if any(gt in r or r in gt for r in retrieved_norm))
    recall_at_5 = hits / len(gt_norm) if gt_norm else 0.0

    # MRR: 1/rank of the first ground-truth doc in the retrieved list
    mrr = 0.0
    for rank, r in enumerate(retrieved_norm, start=1):
        if any(gt in r or r in gt for gt in gt_norm):
            mrr = 1.0 / rank
            break

    # Log both metrics; LangSmith stores the primary key score
    # Return Recall@5 as primary; comment includes MRR
    return {
        "key": "retriever_recall_at_5",
        "score": recall_at_5,
        "comment": f"Recall@5={recall_at_5:.2f}, MRR={mrr:.3f} | "
                   f"GT={gt_norm} | Retrieved={retrieved_norm}"
    }

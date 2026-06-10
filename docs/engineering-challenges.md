# Engineering Challenges

Three non-trivial issues were discovered and fixed during development. Each is documented with what happened, how it was diagnosed, and how it was fixed.

---

## Challenge 1 — LLM JSON Generation Loop

### What happened

The 8B DocGrader model, when asked to grade 6 retrieved documents, occasionally got stuck in an infinite JSON array loop — generating hundreds of `{"relevant": "no"}` entries instead of stopping after 6. This caused a Groq 400 Bad Request error and the DocGrader crashed.

### How it was diagnosed

The Groq API error response includes a `failed_generation` field showing the raw LLM output. Inspecting it revealed the model had generated 300+ array items for a 6-item request. The loop only occurred under rate-limit pressure — the retry logic was hitting an already-overloaded context window, and the model lost track of its stopping condition when re-invoked mid-generation.

### How it was fixed

The prompt now dynamically injects an explicit item count before the documents:

```
CRITICAL INSTRUCTION: There are EXACTLY 6 documents below.
Your JSON array MUST contain EXACTLY 6 items.
Stop generating immediately after the 6th item.
```

### The lesson

Small LLMs need explicit output length constraints in structured generation tasks. The anchor count acts as a hard stopping signal that larger models infer naturally from context but smaller models need stated explicitly. This is especially true when the model is being asked to produce a structured list whose length is variable and input-dependent.

### Fallback behavior

Even when the crash occurred before the fix, the pipeline did not halt. The DocGrader's error boundary defaulted to accepting all documents as relevant — degrading gracefully to Traditional RAG behavior for that one question rather than failing the entire benchmark run. This fallback is the correct design: a retrieval grader failure should degrade quality, not terminate the system.

---

## Challenge 2 — Silent Evaluator Zeroes

### What happened

The LLM-as-a-Judge was occasionally returning `"3"` (a string) instead of `3` (an integer) for scored metrics. Pydantic's structured output validation crashed, and the evaluator silently assigned `score=0.0` to that question — making a perfectly correct answer appear as a complete failure in LangSmith.

### How it was diagnosed

The Faithfulness average appeared suspiciously low for questions where the system logs showed clearly correct responses with full policy grounding. Checking the LangSmith run logs revealed `expected integer, but got string` in the `failed_generation` field of the evaluator's API response. The silent zeroes were invisible in the aggregate metrics — they looked like actual failures, not instrumentation bugs.

### How it was fixed

Two changes were made:

1. Every evaluator prompt now explicitly instructs: *"Output the score as a raw integer (e.g. `2`), NOT a string (e.g. `'2'`)."*
2. Every evaluator's `except` block logs the full error with `logger.error()` so crashes are immediately visible in the terminal output — no more silent zeroes contaminating the aggregate scores.

### The lesson

LLM-as-a-Judge pipelines need the same defensive programming discipline as production APIs. Silent failures in an evaluation pipeline corrupt all downstream conclusions — a benchmark result is only meaningful if you can trust that every data point in it was produced correctly. Treat evaluation code as production code.

---

## Challenge 3 — Biased Completeness Rubric

### What happened

The Completeness evaluator was scoring `1/2` when the agent correctly answered one part of a multi-intent question and safely escalated the unanswerable part. For example, *"I can process your return within 30 days, but I don't have verified information about PS5 availability — please contact support@shopease.com"* was being penalized as incomplete.

### How it was diagnosed

Completeness scores were consistently exactly 0.5 (1/2 normalized) for questions where the system logs showed perfectly correct partial-answer + escalation behavior. Cross-referencing the LangSmith trace with the rubric revealed the evaluator was applying a rubric designed for single-intent questions to multi-intent questions with deliberate partial escalations. The system was being punished for doing exactly what it was designed to do.

### How it was fixed

The rubric was rewritten to distinguish between ignoring a sub-question and explicitly escalating it:

**Old rubric:**
> `2` — Every sub-question fully addressed
> `1` — Primary intent answered; secondary intent missed

**New rubric:**
> `2` — Every sub-question fully addressed, **OR** the agent answered what it could and explicitly escalated the parts it lacked information for. A safe, explicit escalation is a perfect completion of the agent's duty.
> `1` — Primary intent answered, but the agent completely ignored a secondary intent without escalating it.

The corrected rubric was applied to both systems equally, and the benchmark was re-run from scratch.

### The lesson

Evaluation rubrics must be designed to reward the behavior the system was built to exhibit. An agent designed to escalate safely should not be penalized for escalating safely. Rubric design requires the same care as prompt engineering — and it must be validated against actual system outputs before being used to produce conclusions.

---

← [Back to README](../README.md)

# Key Design Decisions

---

## 1. `thought_trace` uses `Annotated[List[dict], operator.add]`

Each graph node returns **only its new trace entry** as a list. LangGraph's `operator.add` reducer automatically accumulates all entries across nodes into a single ordered list by the time the graph reaches its terminal state.

Without this annotation, the default LangGraph state behavior is last-write-wins: the final node's trace entry would overwrite all earlier entries, silently dropping the Router's decision, the Retriever's document list, the DocGrader's pass/fail breakdown, and the HallucinationGrader's verdict. This is a hard-to-debug data loss bug — the graph would still return a response, but the entire thought trace visible in the Chainlit UI would be missing. The `Annotated` reducer is the correct solution: each node appends to the trace independently, and the state machine handles aggregation automatically.

---

## 2. Two-tier LLM routing with 70B budget guard

Groq's free tier caps `llama-3.3-70b-versatile` at **1,000 requests/day**.

| Node | Model | Rationale |
|---|---|---|
| Router | 8B | Binary classification — 8B is sufficient |
| DocGrader | 8B | Per-document yes/no — pattern matching |
| QueryRewriter | 8B | Lexical transformation — no user-facing output |
| HallucinationGrader | 8B | Fact-checking against explicit documents |
| Generator | 70B | User-facing output — quality matters |
| DirectResponder | 8B | Chitchat — low stakes, short output |

The 70B model is reserved exclusively for the Generator: the only node producing a response the customer will actually read. Every other node performs a classification or transformation task where 8B quality is sufficient and the higher throughput of the 8B tier (14,400 RPD) is more valuable than marginal quality improvement. The system logs a warning at 80% daily 70B usage and automatically falls back to 8B at the limit — ensuring the system never crashes at the rate cap, it simply degrades gracefully.

---

## 3. Isolated judge API key for evaluation

`GROQ_API_KEY_JUDGE` is a separate credential from the inference key (`GROQ_API_KEY`). This prevents the evaluation framework's LLM-as-Judge calls from consuming the production inference budget during benchmark runs — they operate against completely independent rate limits.

A 50-question benchmark with 6 evaluators per question and 2 systems requires 600 LLM judge calls. Running this on the same API key as the production system would exhaust the 70B daily quota mid-benchmark, corrupting both the benchmark results and the system's availability to answer real questions. The isolated key is a clean boundary between evaluation infrastructure and production infrastructure.

---

## 4. Traditional RAG baseline for honest comparison

`src/core/naive_rag.py` implements a single-prompt Traditional RAG with **identical guardrails** to the Adaptive system: off-topic detection, adversarial refusal, and escalation when source documents don't contain the answer.

A naive comparison would give Traditional RAG no safety instructions, letting it hallucinate freely — and then claim the Adaptive system's safety improvements as architectural wins when they are actually prompt wins. The current design ensures that any observed difference in benchmark scores reflects a genuine architectural difference: the self-healing loops, the HallucinationGrader, and the DocGrader filtering. Every metric is a relative comparison on equal terms.

---

## 5. Raw ChromaDB over `langchain-chroma`

Using `chromadb` directly rather than LangChain's abstraction wrapper (`langchain-chroma`) gives full control over three critical capabilities:

- **Embedding injection**: Custom BGE-small embeddings are passed directly — no adapter layer needed
- **Distance metric selection**: Cosine similarity is set explicitly at collection creation time
- **Query result structure**: `include=["documents", "metadatas", "distances"]` is specified explicitly, ensuring similarity scores are always available for retrieval observability in the thought trace

The LangChain wrapper hides these controls behind defaults that are not always appropriate. The raw client adds ~20 lines of boilerplate but eliminates hidden behavior and makes every retrieval parameter explicit and auditable.

---

## 6. Scale context for the architecture

The Adaptive self-healing architecture earns its complexity at **200+ document knowledge bases** where:

- Retrieval quality degrades and DocGrader filtering becomes essential — irrelevant chunks are more common when the vector space is crowded
- Multi-document cross-reference queries are the norm rather than the exception
- Query rewriting is required to bridge natural language phrasing and technical policy vocabulary

At the current demo knowledge base of ~30 documents, Traditional RAG with a well-crafted 70B prompt is a credible competitor — and the benchmark results reflect this. The Adaptive system's Faithfulness advantage (+19.3%) is real and meaningful, but the Helpfulness gap (-10.0%) shows that strict guardrails cost more at small scale. The architecture is the correct foundation for production — not over-engineering for a demo. The benchmark validates both the safety benefit and the expected cost, at the scale where the cost/benefit ratio is most challenging.

---

## 7. Iterative Benchmark-Driven Development

The system was not built once and evaluated at the end. Every prompt change (DocGrader leniency rule), rubric correction (Completeness evaluator), and retrieval parameter change (top-k from 5 to 6) was followed by a full LangSmith benchmark re-run to measure the exact impact on all 6 metrics.

This is the same hypothesis > change > measure > iterate feedback loop used in production ML systems — and it was the mechanism by which the Helpfulness score was recovered via leniency prompting without sacrificing Faithfulness. Without the benchmark as a regression harness, the DocGrader leniency change could have been applied blindly — improving some metrics while silently degrading others. The benchmark made the trade-offs visible and quantified. Every design decision in this document was validated against a benchmark number, not just intuition.

---

[Back to README](../README.md)

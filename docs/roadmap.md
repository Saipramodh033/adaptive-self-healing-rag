# Enterprise Roadmap

Each roadmap item is tied to evidence from the benchmark — this is a data-driven roadmap, not a wishlist.

---

| Roadmap Item | Evidence from Benchmark | Expected Impact |
|---|---|---|
| **Hybrid Search (BM25 + dense vector)** | Retriever Recall@6 = 92.7% — keyword-heavy queries (exact product names, city names) miss exact noun matches in pure vector search | Close the remaining 7.3% recall gap; most direct path to higher Helpfulness |
| **Expand KB to 60+ documents** | 5 `missing_info` questions forced escalation even for topics that ShopEase almost certainly has policies on (affiliate programs, Bangalore stores) | Reduce unnecessary escalations; target Helpfulness ≥ 0.90 |
| **Multi-query retrieval (3 variants)** | DocGrader passed only 2–3/6 documents for complex queries even after the leniency rule — the retriever isn't surfacing enough relevant chunks | More retrieval surface area per question without increasing generation token cost |
| **Upgrade embedding model** | `bge-small-en-v1.5` is 384-dim — misses semantic nuance in complex, policy-heavy questions | Higher Recall@6 with zero infrastructure changes; drop-in replacement |
| **Conversation memory (multi-turn)** | System is currently stateless — each query is independent; follow-up questions force re-escalation of context from prior turns | Resolves follow-up questions naturally; critical for real support interactions |
| **Production LLM provider (Azure OpenAI / AWS Bedrock)** | Groq free tier caps 70B at 1,000 RPD — benchmark runs required 8-second sleep intervals between questions to avoid rate limits | Parallel evaluation, production SLAs, no throughput ceiling |
| **Conversation analytics dashboard** | Currently no visibility into which question categories escalate most, which KB documents are retrieved most, or where the DocGrader over-filters | Data-driven KB expansion and prompt tuning decisions |

---

[Back to README](../README.md)

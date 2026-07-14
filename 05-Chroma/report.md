**Task:** Ingest 3 real PDFs into a persistent Chroma store, split with `RecursiveCharacterTextSplitter` at **300 / 500 / 1000** characters (10–20% overlap), and compare retrieval quality for the *same* question across all three chunk sizes.
 
**Corpus:** `high_traffic_apis.pdf`, `container_orchestration.pdf`, `message_queues.pdf` (4 pages each).
 
**Method:** For each query, retrieve top-k from each chunk-size collection and inspect (a) completeness of the retrieved text, (b) redundancy across the top-k, and (c) similarity/distance scores. Lower Chroma distance = closer match.
 
---
 
## Finding 1 — Coverage query: "What techniques protect against a cache stampede?"
 
This is a **breadth** question — the correct answer lists several distinct techniques (locking, jittered expiration, probabilistic early expiration). It tests whether the retrieved chunks collectively cover the full answer.
 
### What was observed
 
| Chunk size | Best distance | Completeness of chunks | Redundancy in top-k |
|-----------|--------------|------------------------|---------------------|
| **300** | **0.1286** (lowest) | Poor — most chunks end mid-word (`"commonly app"`, `"or pro"`) | **High** — chunks #29/#30/#41 overlap heavily, repeating the same 1–2 techniques |
| **500** | 0.1363 | Mixed — one chunk cut mid-sentence; one chunk was *only* the summary line, stripped of technique names | Moderate |
| **1000** | 0.1467 (highest) | **Best** — 3 chunks covered 3 genuinely different techniques + summary | **Low** — each chunk carried a distinct concept |
 
### Observations that cannot be ignored
 
- **Lower distance ≠ better retrieval.** The 300-char collection produced the *lowest* (best-looking) distance score but the *least usable* result. The 1000-char collection had the highest distance yet delivered the most complete, least redundant evidence. **Similarity score and end-usefulness are not the same metric** — this is the single most important takeaway from Day 5.
- **Small chunks fragment concepts.** At 300 chars, chunks routinely end mid-word, so a single technique's explanation is split across two chunks. Answering correctly then depends on *both* fragments being retrieved together, which is not guaranteed.
- **Overlap compounds redundancy at small sizes.** The 10–20% overlap setting, combined with small chunks, produced near-duplicate retrievals (the same sentence sliced three ways). Small chunk size did **not** deliver its theoretical "more diversity per top-k" benefit here — it delivered *repetition* instead.
- **1000 chars won this query** on completeness and coverage, despite ranking worst on raw distance.
---
 
## Finding 2 — Trap query: "How do you configure a visibility timeout in Redis?"
 
This query is **deliberately unanswerable**: "visibility timeout" is a message-queue concept (`message_queues.pdf`); "Redis" only appears as a *caching* example (`high_traffic_apis.pdf`). No chunk in the corpus discusses both together, because that combination does not exist in the source material.
 
### What was observed
 
| Chunk size | Top hits | TTL/caching distractor rank |
|-----------|----------|----------------------------|
| **300** | Visibility-timeout chunks from `message_queues.pdf` (dist 0.187) | TTL chunk from `high_traffic_apis.pdf` ranked lower at 0.2027 |
| **500** | Same — queue chunks on top (dist 0.188) | TTL distractor at 0.2120 |
| **1000** | Same — queue chunk on top (dist 0.189) | Caching chunks at 0.216 / 0.233 |
 
### Observations that cannot be ignored
 
- **Retrieval was NOT fooled by the keyword "Redis."** Despite the query literally containing "Redis," the embedding correctly ranked *message-queue* visibility-timeout content above *caching* content across all three chunk sizes. The system retrieved on **meaning, not keyword overlap** — a genuine positive result, and evidence that dense retrieval behaves differently from keyword search.
- **The caching distractor is consistently present but consistently lower-ranked** (~0.20–0.23 vs ~0.19). Retrieval ordering is correct; the wrong-domain chunk never displaces the right-domain one.
- **The real risk sits at the generation step, not retrieval.** Because no chunk actually answers the question, the generation step must be tested separately for one of three outcomes:
  1. **Faithful (score 5):** model states the context describes queue visibility timeouts, not Redis, and refuses to fabricate.
  2. **Silent blend:** model applies queue content to Redis, producing a plausible but fabricated answer.
  3. **Ungrounded:** model ignores context entirely and answers from pretrained Redis knowledge (`EXPIRE`/TTL) without flagging it isn't in the documents.
- **Action:** this query is the strongest failure-mode probe in the set. Its *generated answer* (not just retrieval) must be scored in the Day 6 naive-RAG exercise. **[OPEN ITEM — generation output not yet captured.]**
---
 
## Consolidated conclusions
 
1. **Chunk size is a breadth-vs-completeness trade-off, not a "best value" to find.**
   - Smaller chunks → sharper embeddings, lower distance scores, but fragmented concepts and (with overlap) redundant top-k.
   - Larger chunks → more self-contained, more complete answers, but fewer distinct chunks fit in a fixed context/token budget.
2. **Do not rank chunk sizes by distance score alone.** On the coverage query, the lowest-distance configuration (300) was the least useful. Judge by whether the retrieved set actually contains a complete, non-redundant answer.
3. **Default choice for this corpus: 1000 chars.** These documents explain concepts in full paragraphs where the claim and its justification sit together; larger chunks preserve that. A corpus of short, self-contained facts (e.g., a glossary or FAQ) could reasonably favor smaller chunks — the right size is corpus-dependent, and this conclusion should not be generalized beyond documents of this shape.
4. **Overlap needs re-tuning at small chunk sizes.** The current 10–20% overlap produced near-duplicate retrievals at 300 chars. If small chunks are used later, overlap should be reduced, or a de-duplication step added to the retriever.
---
 
## Open items carried forward
 
- [ ] Capture and faithfulness-score the **generated** answer for the Redis/visibility-timeout trap query (Day 6).
- [ ] Day 4 benchmark rerun on **clustered** synthetic data (uniform-random data produced 0/5 and 2/5 recall — unresolved).
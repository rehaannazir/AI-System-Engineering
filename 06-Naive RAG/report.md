# Day 6 — Naive RAG Findings

**Task:** Build a hand-rolled retrieve → augment → generate pipeline (no LangChain) on top of the Day 5 Chroma collection. Run 10 questions through it, score faithfulness 1–5 per answer, and identify at least 2 concrete failure modes.

**Pipeline:** query → Chroma top-k retrieval → context-stuffed prompt with citation instruction (metadata injected: `source`, `page`, `chunk_index`, `author`) → Gemini → answer + citations.

**Status: IN PROGRESS — 4 of 10 questions run. Not yet ready to close out.**

---

## Results so far

| # | Question | Answer faithful to context? | Faithfulness (1–5) | Notes |
|---|----------|------------------------------|---------------------|-------|
| 1 | Why might a team avoid running a database as a StatefulSet? | Yes | *(not yet scored — assign before closing)* | Accurate, complete |
| 2 | Liveness probe vs. readiness probe? | Yes | *(not yet scored)* | Captured full nuance, no fragmentation |
| 3 | Horizontal vs. vertical scaling? | Yes | *(not yet scored)* | Accurate, complete |
| 4 | What autoscaling metric does RabbitMQ use by default? | N/A — correctly refused | *(not yet scored — refusal should still get a score, e.g. 5 for correctly saying "I don't know")* | No chunk in corpus answers this; model did not fabricate |

**Note:** numeric scores are placeholders. Faithfulness scoring was discussed but not actually assigned to these four answers yet — this needs to happen before Day 6 can be marked complete.

---

## Failure mode 1 (confirmed): Citation format is not enforced by the prompt

Same prompt template, same available metadata (`source`, `author`), produced three different citation renderings across three answers:

- Q1: `(Source: container_orchestr.pdf)` — author omitted
- Q2: `` `container_orchestr.pdf` by Rehan Nazir `` — author included
- Q3: `Rehan Nazir's 'container_orchestr.pdf'` — author included, different phrasing again

**This is not a hallucination** — the author metadata is legitimately injected into the prompt, so "by Rehan Nazir" is grounded, not invented. The actual problem is **prompt-adherence drift**: the model treats the citation instruction as a loose suggestion rather than a fixed format, and silently drops a field (author) in one of three otherwise-identical cases.

**Also unresolved:** none of the four citations surface `page` or `chunk_index`, even though both are present in the ingested metadata per the Day 5 spec. Without them, an answer can't be traced back to a specific chunk — which defeats the debugging purpose metadata was added for.

**Fix to apply and re-test:**
```
Cite every fact using EXACTLY this format, with no variation:
[Source: <filename> | page <page> | chunk <chunk_index>]

Example: [Source: container_orchestr.pdf | page 3 | chunk 12]
```
Rerun Q1–Q3 after this change and confirm all three fields appear consistently.

---

## Failure mode 2: OPEN — not yet tested

The RabbitMQ query (Q4) is the *easy* version of a hallucination trap: retrieval likely returned nothing resembling an answer, so refusal cost the model nothing.

The harder, unresolved test is the **Redis / visibility-timeout query**:
> "How do you configure a visibility timeout in Redis?"

Day 5 retrieval testing already showed that for this query, the retrieved chunks are **confident, on-topic-looking, real content** — correct queue-related material about visibility timeouts — just applied to the wrong subject (Redis). This gives the model plausible material to blend into a fabricated-but-convincing answer, unlike Q4 where there was nothing to blend.

**Q4 passing does not predict this will pass.** This query has not been run through the generation step yet. It remains the single most important untested case for Day 6, because it's the only query in the set where a hallucination could plausibly hide inside a confident-sounding, well-cited answer.

---

## What's required to close out Day 6

- [ ] Run the remaining 6 questions (currently 4/10 complete)
- [ ] Assign actual faithfulness scores (1–5) to all 10, including the 4 already run
- [ ] Run and score the Redis/visibility-timeout trap query — this is the priority, not just another item on the list
- [ ] Apply the strict citation-format fix above and confirm `page` + `chunk_index` appear in output, then re-verify Q1–Q3 citations are consistent
- [ ] Write final 2 (or more) failure modes with concrete before/after examples, once the above is done

---

## Carried-forward open items (not Day 6, do not drop)

- [ ] Day 4 — clustered synthetic data rerun (uniform-random data produced 0/5 and 2/5 recall)
- [ ] Day 5 — Redis trap query's generated output, which now overlaps directly with the Day 6 item above
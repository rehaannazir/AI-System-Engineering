# Results — Chroma Chunk-Size Experiment

**Query:** `"How do message queues help with high traffic APIs?"`

Corpus: `container_orchestr.pdf`, `high_traffic_apis.pdf`, `message_queues.pdf` — extracted with PyMuPDF (`fitz`), split into overlapping character chunks, embedded with `gemini-embedding-001` (`SEMANTIC_SIMILARITY` task type), stored in a persistent Chroma collection (`hnsw:space: cosine`), retrieved top-3 nearest neighbors.

`ingest.py` was re-run with `to_chunks(text, chunk_size=...)` set to **300**, then **500**, then **1000** (overlap fixed at 100), and `query.py` was re-run after each.

## Raw Results

| chunk_size | Rank | Source | chunk_index | Distance |
|---|---|---|---|---|
| 300 | 1 | high_traffic_apis.pdf | 0 | 0.1453 |
| 300 | 2 | message_queues.pdf | 0 | 0.1462 |
| 300 | 3 | message_queues.pdf | 17 | 0.1556 |
| 500 | 1 | high_traffic_apis.pdf | 0 | 0.1453 |
| 500 | 2 | message_queues.pdf | 0 | 0.1462 |
| 500 | 3 | message_queues.pdf | 17 | 0.1556 |
| 1000 | 1 | high_traffic_apis.pdf | 0 | 0.1453 |
| 1000 | 2 | message_queues.pdf | 0 | 0.1462 |
| 1000 | 3 | message_queues.pdf | 17 | 0.1556 |

Same sources, same `chunk_index` values, same distances to **four decimal places**, same chunk text printed to the console — for three supposedly different chunk sizes.

## What this actually shows (and what it doesn't)

At first glance this looks like "retrieval is robust to chunk size." It isn't — this is a pipeline artifact, not a robustness result. The identical output is explained by how `ingest.py` and Chroma's `PersistentClient` interact, not by chunking having no effect on the embeddings.

**The mechanism:**

1. `chrom = chromadb.PersistentClient("./chroma.db")` points at the same on-disk store across every run — nothing resets it between the 300/500/1000 experiments.
2. `ids.append(f"{source}_{i}")` in [ingest.py:25](05-Chroma/ingest.py#L25) is **deterministic and chunk-size-independent** — chunk 0 of `message_queues.pdf` is always id `message_queues.pdf_0`, regardless of whether chunk_size is 300 or 1000.
3. `collection.add(...)` ([ingest.py:33](05-Chroma/ingest.py#L33)) does not overwrite existing ids. Chroma treats `add()` as insert-only: if an id already exists in the collection, that call is a silent no-op for that id — the original embedding and document text stay exactly as they were on first insert.
4. `get_or_create_collection(name="Docs", ...)` ([ingest.py:29](05-Chroma/ingest.py#L29)) means every re-run reattaches to the *same* collection instead of starting fresh.

So the very first `ingest.py` run (whatever chunk_size was live at the time) populated ids like `high_traffic_apis.pdf_0`, `message_queues.pdf_0`, `message_queues.pdf_17`. Every subsequent re-run — with chunk_size changed to 300, then 500, then 1000 — tried to re-add content under those same ids and was ignored. `query.py` was therefore always searching against embeddings from the *original* ingest, never the ones the later chunk_size values were supposed to produce.

This is also why the result is a strong signal rather than a coincidence: `message_queues.pdf_17` existing in *all three* runs is very unlikely to happen by chance if chunking had actually changed. With `chunk_size=300`, `overlap=100`, chunk 17 sits around character ~4,000; with `chunk_size=1000`, chunk 17 sits around character ~15,300 — those are different regions of the document. Getting the same index *and* the same chunk text *and* the same distance to 4 decimal places across both is only possible if the underlying stored chunk never changed at all.

## Workflow, as the code actually runs it

**Ingestion (`ingest.py` + `helper.py`):**

```
for each PDF in sources:
    extract_text()        → fitz opens the PDF, concatenates page.get_text() for all pages
    to_chunks()            → sliding window: chunks[i] = text[start : start+chunk_size]
                              start advances by (chunk_size - overlap) each step
    → documents[], metadatas=[{source, chunk_index}], ids=[f"{source}_{i}"]

embed_texts(all documents)  → one Gemini embed_content call per chunk,
                               task_type=SEMANTIC_SIMILARITY, 1s sleep between calls (rate limiting)

chrom.get_or_create_collection("Docs", hnsw:space=cosine)
collection.add(ids, documents, embeddings, metadatas)
    → new ids inserted; ids that already exist in "Docs" are skipped
```

**Query (`query.py`):**

```
chrom.get_or_create_collection("Docs", ...)   → reattaches to the same persistent collection
query_embed = embed_texts([query])[0]          → same embedding model/task_type as ingestion
collection.query(query_embeddings=[query_embed], n_results=3)
    → Chroma's HNSW index does approximate nearest-neighbor search over cosine distance
for each of the 3 hits: print source, chunk_index, distance, chunk text
```

Nothing in this flow ever deletes or replaces prior data — `./chroma.db` is purely additive across runs unless ids collide *and* an explicit `upsert` is used instead of `add`.

## What a real chunk-size comparison would need

To actually measure how chunk_size affects retrieval, the collection must not be shared/stale across trials. Options, in order of simplicity:

1. **Fresh collection per trial** — `chrom.delete_collection("Docs")` before each `get_or_create_collection` call, or name the collection per chunk_size (`Docs_300`, `Docs_500`, `Docs_1000`) so trials don't collide.
2. **`collection.upsert()` instead of `collection.add()`** — upsert overwrites existing ids with new embeddings/documents, so re-ingesting with a new chunk_size actually replaces the old chunks (works only if the *number* of chunks per source doesn't shrink between runs, otherwise stale high-index ids from a smaller chunk_size run linger).
3. **Wipe `./chroma.db` on disk** between trials — bluntest, but guarantees no leftover ids.

## Takeaway

The experiment as run doesn't demonstrate that retrieval is chunk-size-invariant for this query — it demonstrates that `ingest.py`'s combination of deterministic ids, a persistent collection, and insert-only `add()` makes repeated ingestion runs silently idempotent after the first one. The "300 vs 500 vs 1000" comparison never actually reached the query step with different data. Rerunning with collection reset (option 1 or 2 above) is required before any conclusion about chunk_size's effect on retrieval quality can be drawn.

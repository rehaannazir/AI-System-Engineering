# Retrieval-Augmented Generation (RAG)

## 1. Definition
RAG is an architecture that combines a retriever with a generator: instead of
relying only on an LLM's parametric memory, relevant documents are fetched
from an external knowledge base at query time and injected into the prompt
as context. RAG was first proposed by Lewis et al. in 2020 (Facebook AI
Research).

## 2. Core Pipeline Stages

### 2.1 Ingestion
- Load raw documents (PDF, HTML, TXT, Markdown, etc.)
- Split into chunks, typically 200-1000 tokens with 10-20% overlap
- Embed each chunk into a dense vector using a model such as
  `text-embedding-3-small` or `all-MiniLM-L6-v2`
- Store vectors in an index (FAISS, Chroma, Pinecone, Weaviate)

### 2.2 Retrieval
- Embed the user query with the same embedding model
- Run a similarity search (cosine similarity or dot product) to fetch the
  top-k chunks (commonly k = 3 to k = 10)
- Optional: apply a re-ranker (e.g., Cohere Rerank, cross-encoder) to reorder
  candidates by relevance

### 2.3 Generation
- Concatenate retrieved chunks into the LLM's context window
- Prompt the LLM to answer strictly using the provided context
- Return the answer along with citations to the source chunks

## 3. Evaluation Metrics
- **Context Precision** — fraction of retrieved chunks that are relevant
- **Context Recall** — fraction of relevant chunks that were retrieved
- **Faithfulness** — whether the generated answer is grounded in context
- **Answer Relevance** — whether the answer actually addresses the query
- **Latency** — end-to-end time from query to answer (retrieval + generation)

## 4. Common Failure Modes
1. **Chunking too coarse** → irrelevant text dilutes the context
2. **Chunking too fine** → loses surrounding context, breaks semantic units
3. **Embedding mismatch** — different models used for indexing vs. querying
4. **Stale index** — knowledge base not updated after source documents change
5. **Lost-in-the-middle** — LLMs attend less to context placed in the middle
   of a long prompt

## 5. Variants
| Variant | Key Idea |
|---|---|
| Naive RAG | Single retrieve-then-generate pass |
| Hybrid RAG | Combines dense vector search with sparse keyword search (BM25) |
| Multi-Query RAG | Generates multiple query rewrites to widen recall |
| Graph RAG | Retrieves from a knowledge graph instead of / alongside vectors |
| Agentic RAG | An agent decides iteratively whether to retrieve again |

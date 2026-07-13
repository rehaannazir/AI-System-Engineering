# Results

**Query:** `"I love to do coding in silence"`

Dataset: 15 sentences embedded with `gemini-embedding-001` (`SEMANTIC_SIMILARITY` task type), searched for top-3 nearest neighbors.

| Index Type | Top-3 Indexes | Distances |
|---|---|---|
| `IndexFlatL2` (exact) | 0, 13, 8 | 0.5428, 0.5487, 0.5678 |
| `IndexIVFFlat` (nlist=3, nprobe=1) | 0, 13, 8 | 0.5428, 0.5487, 0.5678 |
| `IndexHNSWFlat` (M=5) | 0, 13, 8 | 0.5428, 0.5487, 0.5678 |

**Matched sentences (same for all three methods, in rank order):**

| Rank | Index | Sentence | Distance |
|---|---|---|---|
| 1 | 0 | "Artificial Intelligence enables computers to perform tasks that normally require human intelligence" | 0.5428 |
| 2 | 13 | "Consistent exercise reduces stress and improves mental well-being" | 0.5487 |
| 3 | 8 | "Many successful people read books daily to continue learning and growing" | 0.5678 |

## Observations

All three index types — exact (`IndexFlatL2`), inverted-file (`IndexIVFFlat`), and graph-based (`IndexHNSWFlat`) — returned **identical top-3 results and distances** for this query. This is expected at this scale, not evidence that IVF/HNSW are always exact:

- **Dataset is tiny (15 vectors).** `IndexFlatL2` runs a brute-force scan, so it's always the ground truth. With `nlist=3` and `nprobe=1`, IVF still searches roughly a third of the dataset per query — easily enough to find the true top-3 when there are only 15 vectors total. Likewise, HNSW's graph is nearly fully connected at this size, so its greedy search converges to the exact answer.
- **The approximation gap only shows up at scale.** IVF and HNSW trade recall for speed by searching a subset of the index (a few clusters, or a bounded graph walk) instead of every vector. That trade-off is invisible here because "a subset of 15" is still almost everything. At millions of vectors, `nprobe`/`efSearch` tuning starts to matter — too low, and IVF/HNSW will diverge from the exact `IndexFlatL2` result in exchange for much faster search.
- **None of the top matches are strong semantic hits** for a query about "coding in silence" — the closest sentences are about AI/computers, exercise, and reading habits, none of which really discuss coding. This reflects the small, topically mixed corpus (15 sentences spanning AI, health, and books) rather than a flaw in the indexes themselves — there's simply no sentence in the dataset that's a close semantic match to the query.

## 200-Sentence Trade-off Comparison

**Query:** `"I love to do coding in silence"`

Dataset: 200 sentences (20 topics, 10 sentences each) embedded with `gemini-embedding-001` (`SEMANTIC_SIMILARITY` task type), searched for top-5 nearest neighbors. `IndexFlatL2` results are treated as ground truth for recall.

| Index | Corpus Size | Recall | Latency (ns) | Reason |
|---|---|---|---|---|
| `IndexFlatL2` (exact) | 200 | 5/5 | 5,664,587 | Brute-force scan over all vectors — always exact, so it's the ground truth; cost grows linearly with corpus size. |
| `IndexIVFFlat` (nlist=20, nprobe=4) | 200 | 4/5 | 6,237,507 | Missed index `198` (found `172` instead) because 200 points split into 20 clusters gives ~10 points/cluster — too few for FAISS's k-means to place centroids well (`"please provide at least 780 training points"` warning). Also slower than Flat here because with so few vectors, the overhead of quantizer lookup + per-list search outweighs any savings from skipping clusters. |
| `IndexHNSWFlat` (M=30) | 200 | 5/5 | 2,774,715 | Matched Flat exactly and was the fastest — at only 200 vectors the HNSW graph is small and densely connected, so the greedy graph walk converges on the true nearest neighbors while still avoiding a full scan. |

**Takeaway:** at this small a corpus, none of these numbers reflect the indexes' real-world behavior — IVF looks *worse* than brute-force only because the dataset is far too small for its clustering to help. The trade-offs these indexes are designed for (sub-linear search time, tunable recall/speed via `nprobe`/`efSearch`) only become visible at tens of thousands to millions of vectors, where Flat's linear scan becomes the bottleneck and IVF/HNSW pull ahead.

## 100,000 x 768 Dummy Vector Benchmark

**Corpus:** 100,000 random vectors (`np.random.random`, uniform in `[0,1)`), dimension 768, searched for top-5 nearest neighbors against one random query vector. `IndexFlatL2` results are treated as ground truth for recall. HNSW's `efSearch` was raised from FAISS's default (16) to **128** to test whether recall recovers.

| Index | Corpus Size | Recall | Latency (ns) | Reason |
|---|---|---|---|---|
| `IndexFlatL2` (exact) | 100,000 | 5/5 | 50,081,491 | Brute-force scan over all 100k × 768 vectors — always exact, but cost scales linearly with corpus size. |
| `IndexIVFFlat` (nlist=100, nprobe=8) | 100,000 | 0/5 | 8,882,999 | Only searches 8 of 100 clusters (~8% of the corpus). None of the true top-5 fell into a visited cluster — with uniformly random 768-dim data, distances barely vary between points, so `nprobe=8` has no reliable signal for which clusters actually hold the nearest neighbors. ~5.6x faster than Flat. |
| `IndexHNSWFlat` (M=32, efSearch=128) | 100,000 | 2/5 | 3,843,546 | Recall improved from 0/5 (at the default `efSearch=16`) to 2/5 after raising `efSearch` to 128, confirming the earlier miss was a tuning issue — and on this run it was also the fastest of the three (~13x faster than Flat), since a bounded graph walk still beats a full linear scan at this corpus size. |

**Takeaway:** raising `efSearch` did exactly what it's supposed to — traded speed for recall, and recall went up, though it's still far from perfect. This reinforces the earlier point: uniform random vectors in 768 dimensions are close to worst-case for ANN search, since distances between points concentrate and there's little real neighborhood structure for IVF's clusters or HNSW's graph to exploit. On real embeddings (like the Gemini-generated ones used above), these same indexes typically reach much higher recall at far lower `nprobe`/`efSearch` settings, because genuine semantic data has actual clusters for these structures to find.

Note: absolute latencies swing run-to-run (no fixed seed, no warm-up, single query) — e.g. an earlier run had Flat as the slowest and HNSW as the slowest, here HNSW comes out fastest. Treat the *recall* numbers as the stable signal and latency as directional only, unless benchmarked with multiple queries/repeats.


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


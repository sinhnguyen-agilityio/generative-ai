# Contract: Passage Lookup

Module: `storage/passage_lookup.py`. Used by the searcher (gap self-lookup before its report) and
by `analyze_gaps` (run-wide lookup of essential gaps).

## Interface

```python
class PassageLookup:
    def __init__(self, index: VectorIndex, *, top_k: int = 5, candidates: int = 10,
                 rrf_k: int = 60): ...

    def lookup(self, queries: list[str], *, run_id: str,
               allowed_source_ids: set[str] | None) -> LookupResult: ...
```

`LookupResult`: `chunks: list[IndexedChunk]` (≤ `top_k`, best first), `ranks:
list[{chunk_id, vector_rank, keyword_rank, fused}]`, `fallback: "none" | "vector_only"`,
`error: str`.

## VectorIndex additions (`storage/vector_index.py`, `ports.py`)

```python
def documents(self, run_id: str) -> list[IndexedChunk]: ...   # every chunk of the run
```

- `ChromaVectorIndex`: `collection.get(where={"run_id": run_id}, include=["documents",
  "metadatas"])`; upserts also store `chunk_index`.
- `LocalVectorIndex`: filter its JSON rows by `metadata.run_id`.

## Algorithm

1. For each query: `index.retrieve(query, limit=candidates, run_id=run_id)` → vector ranking;
   BM25 over `documents(run_id)` (built once per lookup call, lowercase `\w+` tokens, k1 = 1.5,
   b = 0.75) → keyword ranking, top `candidates`.
2. Drop chunks whose `source_id` is not in `allowed_source_ids`. Callers always pass the sources
   of the run's active tasks, so pages captured for a superseded plan are never returned.
3. Fused score = Σ over every ranking that contains the chunk of `1 / (rrf_k + rank)`.
4. Return the `top_k` highest fused chunks; ties broken by vector rank, then chunk id.

## Failure behavior

| Failure | Behavior |
| --- | --- |
| `documents()` unavailable or raises | Vector ranking only; `fallback = "vector_only"` |
| Vector retrieval raises | Keyword ranking only if available, else empty result; `error` set (redacted) |
| No chunks for the run | Empty result, no error |

A lookup never writes to the index and never waits for running tasks.

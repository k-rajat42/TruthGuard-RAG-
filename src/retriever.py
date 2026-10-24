"""Qdrant-native hybrid retriever (dense bge-small + sparse BM25, fused with RRF).

Corpus plug-in point
--------------------
Nothing in this file knows about any particular dataset. Feed it a list of dicts
matching `DocumentChunk`, either directly or via a file:

    r = HybridRetriever()
    r.ingest(list_of_dicts)                 # programmatic
    r.ingest_file("my_corpus.jsonl")        # .json (array) or .jsonl

Add new file formats (e.g. CSV) by registering a reader in `_READERS`.

Supersession ("prefer latest")
------------------------------
`retrieve()` returns the FULL pool (so the Critic/Resolver/UI can show provenance),
annotated with `superseded_by`. `current_chunks()` gives the Synthesizer view with
stale chunks hidden. "Stale" here means an EXPLICIT `supersedes` link inside the pool;
date/authority mismatches without a link are left to the Phase 3 metadata analysis.

CLI test (works with any corpus file):
    python -m src.retriever --corpus my.jsonl --query "What is the stipend?" \
        --expect DOC-HR-01 DOC-HR-02
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import ValidationError
from qdrant_client import QdrantClient, models

from src.config import SETTINGS, Settings
from src.schemas import DocumentChunk, RetrievedChunk


class CorpusError(ValueError):
    """Raised on the FIRST bad record / unreadable corpus file (fail-fast by design)."""


# --------------------------------------------------------------------------- #
# Corpus loading + validation (decoupled from Qdrant)
# --------------------------------------------------------------------------- #
def _read_json(path: Path) -> list[Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise CorpusError(f"{path}: invalid JSON ({e.msg} at line {e.lineno})") from e
    if isinstance(data, dict) and isinstance(data.get("chunks"), list):
        data = data["chunks"]
    if not isinstance(data, list):
        raise CorpusError(f"{path}: expected a JSON array (or an object with a 'chunks' array)")
    return data


def _read_jsonl(path: Path) -> list[Any]:
    records = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise CorpusError(f"{path}:{n}: invalid JSON ({e.msg})") from e
    return records


_READERS: dict[str, Callable[[Path], list[Any]]] = {".json": _read_json, ".jsonl": _read_jsonl}


def load_chunks_from_file(path: str | Path) -> list[dict]:
    """Read a .json / .jsonl corpus into a list of dicts (no schema validation yet)."""
    p = Path(path)
    if not p.is_file():
        raise CorpusError(f"Corpus file not found: {p}")
    reader = _READERS.get(p.suffix.lower())
    if reader is None:
        raise CorpusError(f"Unsupported corpus format '{p.suffix}'. Supported: {sorted(_READERS)}")
    records = reader(p)
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise CorpusError(f"{p}: record #{i} is not a JSON object")
    return records


def validate_chunks(records: list[dict]) -> list[DocumentChunk]:
    """Validate every dict against DocumentChunk. Raises CorpusError on the first bad record.

    Assigns `chunk_index` per doc_id (in input order) when omitted, and rejects
    duplicate (doc_id, chunk_index) pairs within the batch.
    """
    chunks: list[DocumentChunk] = []
    next_index: dict[str, int] = defaultdict(int)
    seen: set[tuple[str, int]] = set()
    for i, rec in enumerate(records):
        try:
            chunk = DocumentChunk.model_validate(rec)
        except ValidationError as e:
            doc = rec.get("doc_id", "?") if isinstance(rec, dict) else "?"
            raise CorpusError(f"Record #{i} (doc_id={doc}) failed validation:\n{e}") from e
        if chunk.chunk_index is None:
            chunk.chunk_index = next_index[chunk.doc_id]
        key = (chunk.doc_id, chunk.chunk_index)
        if key in seen:
            raise CorpusError(f"Record #{i}: duplicate (doc_id, chunk_index) = {key}")
        seen.add(key)
        next_index[chunk.doc_id] = max(next_index[chunk.doc_id], chunk.chunk_index + 1)
        chunks.append(chunk)
    return chunks


# --------------------------------------------------------------------------- #
# Supersession helpers (pure functions, used by retriever + agents)
# --------------------------------------------------------------------------- #
def annotate_supersession(pool: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Fill `superseded_by` using explicit `supersedes` links between chunks in the pool."""
    superseders: dict[str, set[str]] = defaultdict(set)
    for r in pool:
        for old in r.chunk.supersedes:
            if old != r.chunk.doc_id:
                superseders[old].add(r.chunk.doc_id)
    for r in pool:
        r.superseded_by = sorted(superseders.get(r.chunk.doc_id, set()))
    return pool


def current_chunks(pool: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Synthesizer view: the pool with explicitly superseded (stale) chunks hidden."""
    return [r for r in pool if not r.is_superseded]


def superseded_chunks(pool: list[RetrievedChunk]) -> list[RetrievedChunk]:
    return [r for r in pool if r.is_superseded]


# --------------------------------------------------------------------------- #
# Retriever
# --------------------------------------------------------------------------- #
class HybridRetriever:
    def __init__(
        self,
        settings: Settings = SETTINGS,
        path: Optional[str] = None,
        prefetch_limit: Optional[int] = None,
    ):
        self.s = settings
        self.path = path or settings.qdrant_path
        self.prefetch_limit = prefetch_limit or settings.prefetch_limit
        # NOTE: disk-backed local Qdrant allows ONE client per folder at a time.
        self.client = QdrantClient(":memory:") if self.path == ":memory:" else QdrantClient(path=self.path)
        self._ensure_collection()

    # -- collection -------------------------------------------------------- #
    def _ensure_collection(self) -> None:
        if self.client.collection_exists(self.s.collection_name):
            return
        self.client.create_collection(
            collection_name=self.s.collection_name,
            vectors_config={
                self.s.dense_vector_name: models.VectorParams(size=384, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={
                self.s.sparse_vector_name: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )

    def reset(self) -> None:
        """Drop and recreate the collection (use before ingesting a different corpus)."""
        self.client.delete_collection(self.s.collection_name)
        self._ensure_collection()

    def count(self) -> int:
        return self.client.count(self.s.collection_name, exact=True).count

    # -- ingestion --------------------------------------------------------- #
    def ingest(self, chunks: list[dict]) -> int:
        """Validate (fail-fast) then upsert. Idempotent: same doc_id+chunk_index overwrites."""
        validated = validate_chunks(chunks)  # nothing is written if any record is bad
        bs = self.s.ingest_batch_size
        for start in range(0, len(validated), bs):
            points = [
                models.PointStruct(
                    id=c.point_id(),
                    vector={
                        self.s.dense_vector_name: models.Document(text=c.text, model=self.s.dense_model),
                        self.s.sparse_vector_name: models.Document(text=c.text, model=self.s.sparse_model),
                    },
                    payload=c.to_payload(),
                )
                for c in validated[start : start + bs]
            ]
            self.client.upsert(collection_name=self.s.collection_name, points=points, wait=True)
        return len(validated)

    def ingest_file(self, path: str | Path) -> int:
        return self.ingest(load_chunks_from_file(path))

    # -- retrieval --------------------------------------------------------- #
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        prefetch_limit: Optional[int] = None,
        expand_supersession: bool = True,
    ) -> list[RetrievedChunk]:
        """One query_points call: dense + sparse prefetch fused by RRF.

        Returns the full annotated pool. With expand_supersession, chunks linked to a
        retrieved chunk via `supersedes` (either direction) are appended with
        retrieval_source='expanded', so the stale/new pair is never split by ranking.
        The pool may therefore exceed top_k.
        """
        top_k = top_k or self.s.top_k
        limit = max(prefetch_limit or self.prefetch_limit, top_k)
        res = self.client.query_points(
            collection_name=self.s.collection_name,
            prefetch=[
                models.Prefetch(
                    query=models.Document(text=query, model=self.s.dense_model),
                    using=self.s.dense_vector_name,
                    limit=limit,
                ),
                models.Prefetch(
                    query=models.Document(text=query, model=self.s.sparse_model),
                    using=self.s.sparse_vector_name,
                    limit=limit,
                ),
            ],
            query=models.RrfQuery(rrf=models.Rrf(k=self.s.rrf_k)),
            limit=top_k,
            with_payload=True,
        )
        pool = [
            RetrievedChunk(chunk=DocumentChunk.model_validate(p.payload), rrf_score=float(p.score))
            for p in res.points
        ]
        if expand_supersession:
            pool = self._expand_supersession(pool)
        return annotate_supersession(pool)

    def _expand_supersession(self, pool: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """One-hop expansion: fetch docs this pool supersedes, and docs that supersede this pool."""
        if not pool:
            return pool
        have_docs = {r.chunk.doc_id for r in pool}
        have_points = {r.chunk.point_id() for r in pool}
        older = {d for r in pool for d in r.chunk.supersedes} - have_docs

        conditions = [
            models.FieldCondition(key="supersedes", match=models.MatchAny(any=sorted(have_docs)))
        ]
        if older:
            conditions.append(models.FieldCondition(key="doc_id", match=models.MatchAny(any=sorted(older))))

        points, _ = self.client.scroll(
            collection_name=self.s.collection_name,
            scroll_filter=models.Filter(should=conditions),
            limit=100,
            with_payload=True,
        )
        for p in points:
            chunk = DocumentChunk.model_validate(p.payload)
            if chunk.point_id() in have_points:
                continue
            have_points.add(chunk.point_id())
            pool.append(RetrievedChunk(chunk=chunk, rrf_score=0.0, retrieval_source="expanded"))
        return pool

    # -- inspection (Streamlit payload explorer) --------------------------- #
    def list_points(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.s.collection_name, limit=64, offset=offset, with_payload=True
            )
            out.extend({"point_id": str(p.id), **(p.payload or {})} for p in points)
            if offset is None:
                return out


# --------------------------------------------------------------------------- #
# CLI verification (corpus-agnostic)
# --------------------------------------------------------------------------- #
def _print_pool(pool: list[RetrievedChunk]) -> None:
    print(f"{'#':<3}{'doc_id':<16}{'type':<11}{'date':<12}{'ver':<6}{'rrf':<9}{'source':<9}superseded_by")
    for i, r in enumerate(pool, 1):
        c = r.chunk
        print(
            f"{i:<3}{c.doc_id:<16}{c.doc_type:<11}{c.effective_date.isoformat():<12}{c.version:<6}"
            f"{r.rrf_score:<9.4f}{r.retrieval_source:<9}{','.join(r.superseded_by) or '-'}"
        )


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="TruthGuard hybrid retriever test")
    ap.add_argument("--corpus", required=True, help="Path to .json / .jsonl corpus")
    ap.add_argument("--query", required=True)
    ap.add_argument("--expect", nargs="*", default=[], help="doc_ids that MUST appear in the pool")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--prefetch-limit", type=int, default=None)
    ap.add_argument("--no-expand", action="store_true", help="Disable supersession expansion (pure RRF)")
    args = ap.parse_args(argv)

    try:
        r = HybridRetriever(path=":memory:", prefetch_limit=args.prefetch_limit)
        n = r.ingest_file(args.corpus)
    except CorpusError as e:
        print(f"CORPUS ERROR: {e}", file=sys.stderr)
        return 2
    print(f"Ingested {n} chunks ({r.count()} points in Qdrant)\n")

    pool = r.retrieve(args.query, top_k=args.top_k, expand_supersession=not args.no_expand)
    print(f"Query: {args.query!r}\n")
    _print_pool(pool)

    synth_view = [x.chunk.doc_id for x in current_chunks(pool)]
    print(f"\nSynthesizer view (stale hidden): {synth_view}")

    if args.expect:
        got = {x.chunk.doc_id for x in pool}
        missing = [d for d in args.expect if d not in got]
        if missing:
            print(f"\nFAIL: missing from pool: {missing}")
            return 1
        print(f"\nPASS: all expected doc_ids present: {args.expect}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

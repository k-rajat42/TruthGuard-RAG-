"""Data contracts for TruthGuard.

Key distinction every agent must respect:
  * CORPUS CONFLICT  -> two retrieved documents disagree with each other (or one is stale).
                        This is a property of the DOCUMENTS. The Synthesizer is not at fault.
  * HALLUCINATION    -> the Synthesizer's draft asserts something no retrieved document says.
                        This is a property of the DRAFT. Fixable by retrying the Synthesizer.
"""
import uuid
from datetime import date
from typing import Any, Literal, Optional, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_ID_NAMESPACE = uuid.UUID("6f1c2f0e-8c1e-4b7a-9a52-3d7c1b9d1e11")


# --------------------------------------------------------------------------- #
# Corpus contract
# --------------------------------------------------------------------------- #
class DocumentChunk(BaseModel):
    """One ingestible chunk. External corpora must be dicts matching this shape.

    extra="allow": additional keys (title, region, source_url, ...) are kept and
    stored in the Qdrant payload, so richer schemas can be plugged in later
    without changing this model.
    """

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    text: str = Field(min_length=1, description="The chunk's text content.")
    doc_id: str = Field(min_length=1, description="Stable document identifier, e.g. 'DOC-HR-01'. Shared by all chunks of one document.")
    doc_type: str = Field(description="Authority class, e.g. 'policy', 'amendment', 'memo', 'faq'. Lowercased. Ranked via config.DOC_TYPE_HIERARCHY.")
    effective_date: date = Field(description="ISO date (YYYY-MM-DD) the document takes effect.")
    version: str = Field(description="Document version label, e.g. '1.0'. Coerced to string.")
    supersedes: list[str] = Field(default_factory=list, description="doc_ids this document explicitly replaces. Accepts None, a single string, or a list.")
    topic: Optional[str] = Field(default=None, description="Optional subject tag (e.g. 'remote_work_stipend'). When present, chunks sharing a topic are compared for date/authority mismatches.")
    chunk_index: Optional[int] = Field(default=None, ge=0, description="Position of this chunk within its document. Assigned by the retriever at ingest if omitted.")

    @field_validator("doc_type", mode="after")
    @classmethod
    def _norm_doc_type(cls, v: str) -> str:
        v = v.strip().lower()
        if not v:
            raise ValueError("doc_type must not be empty")
        return v

    @field_validator("version", mode="before")
    @classmethod
    def _coerce_version(cls, v: Any) -> str:
        return str(v)

    @field_validator("supersedes", mode="before")
    @classmethod
    def _coerce_supersedes(cls, v: Any) -> list[str]:
        if v is None or v == "":
            return []
        if isinstance(v, str):
            return [v.strip()]
        return [str(x).strip() for x in v]

    @field_validator("topic", mode="after")
    @classmethod
    def _norm_topic(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip().lower()
        return v or None

    def point_id(self) -> str:
        """Deterministic UUID from doc_id + chunk_index -> idempotent re-ingestion."""
        idx = 0 if self.chunk_index is None else self.chunk_index
        return str(uuid.uuid5(_ID_NAMESPACE, f"{self.doc_id}::{idx}"))

    def to_payload(self) -> dict[str, Any]:
        """JSON-safe payload stored alongside the vectors in Qdrant."""
        return self.model_dump(mode="json")


class RetrievedChunk(BaseModel):
    """A chunk as returned by the hybrid retriever, with its fused RRF score."""

    chunk: DocumentChunk
    rrf_score: float = Field(description="Reciprocal Rank Fusion score from Qdrant. Higher = more relevant. 0.0 for chunks added by supersession expansion.")
    retrieval_source: Literal["rrf", "expanded"] = Field(default="rrf", description="'rrf' = ranked by hybrid search. 'expanded' = pulled in because it supersedes / is superseded by a retrieved chunk.")
    superseded_by: list[str] = Field(default_factory=list, description="doc_ids IN THE POOL that explicitly supersede this chunk's document. Non-empty = stale.")

    @property
    def is_superseded(self) -> bool:
        return bool(self.superseded_by)


# --------------------------------------------------------------------------- #
# Deterministic metadata analysis (Phase 3 pre-critic helper output)
# --------------------------------------------------------------------------- #
class MetadataFlag(BaseModel):
    kind: Literal["supersedes_link", "date_mismatch", "authority_mismatch", "equal_authority_deadlock"] = Field(
        description="supersedes_link: A names B in its supersedes list. date_mismatch: same topic, different effective_date. "
        "authority_mismatch: same topic, different doc_type rank. equal_authority_deadlock: same topic, same date, same rank, no supersedes link."
    )
    doc_ids: list[str] = Field(description="doc_ids involved. For supersedes_link: [newer_doc_id, older_doc_id].")
    topic: Optional[str] = Field(default=None, description="Shared topic, if the flag came from a topic comparison.")
    detail: str = Field(description="One-sentence human-readable explanation.")


# --------------------------------------------------------------------------- #
# Synthesizer
# --------------------------------------------------------------------------- #
class Claim(BaseModel):
    statement: str = Field(description="ONE factual sentence in the answer, ending with its inline [DOC-ID] citation(s).")
    citations: list[str] = Field(min_length=1, description="doc_ids of retrieved chunks that directly state this claim. Never cite a doc_id that was not retrieved.")


class SynthesizerOutput(BaseModel):
    answer: str = Field(description="Final prose answer. Every sentence carries an inline [DOC-ID] citation. Use ONLY facts present in the retrieved chunks.")
    claims: list[Claim] = Field(default_factory=list, description="The answer decomposed into atomic cited claims, in order.")

    @property
    def cited_doc_ids(self) -> list[str]:
        return sorted({d for c in self.claims for d in c.citations})


# --------------------------------------------------------------------------- #
# Critic
# --------------------------------------------------------------------------- #
class ContradictionItem(BaseModel):
    """Two RETRIEVED DOCUMENTS disagree. Never use this for errors in the draft."""

    topic: str = Field(description="What the documents disagree about, e.g. 'monthly remote stipend'.")
    doc_id_a: str = Field(description="First retrieved document's doc_id.")
    doc_id_b: str = Field(description="Second retrieved document's doc_id.")
    position_a: str = Field(description="What document A states (paraphrase, taken from its text).")
    position_b: str = Field(description="What document B states (paraphrase, taken from its text).")


class StaleCitation(BaseModel):
    """The draft relies on a document that another RETRIEVED document supersedes or outdates."""

    claim: str = Field(description="The draft claim that rests on the outdated document.")
    stale_doc_id: str = Field(description="doc_id of the outdated document.")
    superseding_doc_id: str = Field(description="doc_id of the newer / overriding retrieved document.")


class UnsupportedClaim(BaseModel):
    """A HALLUCINATION: the draft asserts something that NO retrieved chunk states."""

    claim: str = Field(description="The draft sentence or fact that has no support.")
    reason: str = Field(description="Why unsupported: fabricated number, wrong citation, or not present in any retrieved chunk.")


class CritiqueReport(BaseModel):
    contradictions: list[ContradictionItem] = Field(default_factory=list, description="Conflicts BETWEEN retrieved documents only.")
    stale_citations: list[StaleCitation] = Field(default_factory=list, description="Draft claims resting on superseded/outdated documents.")
    unsupported_claims: list[UnsupportedClaim] = Field(default_factory=list, description="Hallucinations in the draft. Empty if every claim is backed by a retrieved chunk.")
    has_hallucination: bool = Field(default=False, description="True iff unsupported_claims is non-empty. (Recomputed deterministically.)")
    has_corpus_conflict: bool = Field(default=False, description="True iff contradictions or stale_citations is non-empty. (Recomputed deterministically.)")
    is_clean: bool = Field(default=False, description="True iff no hallucination and no corpus conflict. (Recomputed deterministically.)")
    summary: str = Field(default="", description="Two-sentence explanation of the verdict.")

    @model_validator(mode="after")
    def _derive_flags(self) -> "CritiqueReport":
        # Never trust LLM-set booleans: derive them from the lists so routing is deterministic.
        self.has_hallucination = bool(self.unsupported_claims)
        self.has_corpus_conflict = bool(self.contradictions or self.stale_citations)
        self.is_clean = not (self.has_hallucination or self.has_corpus_conflict)
        return self


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #
class SourcePosition(BaseModel):
    doc_id: str = Field(description="Source document.")
    claim: str = Field(description="What this source says, with its [DOC-ID] citation.")


class ResolutionOutput(BaseModel):
    status: Literal["VERIFIED_CLEAN", "RESOLVED_BY_PROVENANCE", "UNRESOLVABLE_CONFLICT"] = Field(
        description="VERIFIED_CLEAN: no issues. RESOLVED_BY_PROVENANCE: conflict settled via supersedes / effective_date / doc_type hierarchy. "
        "UNRESOLVABLE_CONFLICT: equal date and authority, no supersedes link; do NOT blend."
    )
    final_answer: str = Field(description="User-facing answer with inline [DOC-ID] citations. For UNRESOLVABLE_CONFLICT, present both sides without choosing.")
    winning_sources: list[str] = Field(default_factory=list, description="doc_ids whose position prevails. Must be empty for UNRESOLVABLE_CONFLICT.")
    rejected_sources: list[str] = Field(default_factory=list, description="doc_ids that were superseded or outranked. Must be empty for VERIFIED_CLEAN.")
    positions: list[SourcePosition] = Field(default_factory=list, description="Each conflicting source's stated position (always populated for RESOLVED_BY_PROVENANCE and UNRESOLVABLE_CONFLICT).")
    rules_applied: list[str] = Field(default_factory=list, description="Deterministic adjudication steps, e.g. 'DOC-B over DOC-A: explicit supersedes link'.")
    dropped_unsupported_claims: list[str] = Field(default_factory=list, description="Draft claims still unsupported after max retries; removed from the final answer.")
    rationale: str = Field(description="Why this status was chosen and, if resolved, why the winner prevails (cite the rule used).")

    @model_validator(mode="after")
    def _check_consistency(self) -> "ResolutionOutput":
        if self.status == "UNRESOLVABLE_CONFLICT" and self.winning_sources:
            raise ValueError("UNRESOLVABLE_CONFLICT must not name winning_sources")
        if self.status == "VERIFIED_CLEAN" and self.rejected_sources:
            raise ValueError("VERIFIED_CLEAN must not have rejected_sources")
        if self.status == "RESOLVED_BY_PROVENANCE" and not self.winning_sources:
            raise ValueError("RESOLVED_BY_PROVENANCE requires winning_sources")
        return self


# --------------------------------------------------------------------------- #
# LangGraph state
# --------------------------------------------------------------------------- #
class TruthGuardState(TypedDict, total=False):
    query: str
    retrieved: list[RetrievedChunk]          # hybrid pool from Qdrant
    metadata_flags: list[MetadataFlag]       # deterministic pre-critic output
    draft: SynthesizerOutput                 # latest synthesizer draft
    critique: CritiqueReport                 # latest critic verdict
    critique_feedback: Optional[str]         # fed back to synthesizer on retry
    revision_count: int                      # synthesizer retries so far
    resolution: ResolutionOutput             # final output
    trace: list[dict[str, Any]]              # per-step snapshots for the UI (drafts, critiques)
    # optional retrieval overrides (set by the UI)
    top_k: Optional[int]
    prefetch_limit: Optional[int]
    expand_supersession: bool
    synth_model: Optional[str]
    critic_model: Optional[str]
    resolver_model: Optional[str]

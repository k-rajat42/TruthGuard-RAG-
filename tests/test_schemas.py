import pytest
from pydantic import ValidationError

from src.schemas import CritiqueReport, DocumentChunk, ResolutionOutput, UnsupportedClaim

BASE = {"text": "x", "doc_id": "D1", "doc_type": " Policy ", "effective_date": "2025-03-01", "version": 2}


def test_chunk_normalizes():
    c = DocumentChunk(**BASE, supersedes="D0", topic=" Stipend ")
    assert c.doc_type == "policy" and c.version == "2" and c.supersedes == ["D0"] and c.topic == "stipend"


def test_supersedes_none_and_extras_kept():
    c = DocumentChunk(**BASE, supersedes=None, region="EU")
    assert c.supersedes == [] and c.to_payload()["region"] == "EU"


def test_point_id_deterministic_and_distinct():
    a = DocumentChunk(**BASE, chunk_index=0)
    assert a.point_id() == DocumentChunk(**BASE, chunk_index=0).point_id()
    assert a.point_id() != DocumentChunk(**BASE, chunk_index=1).point_id()


def test_bad_record_fails_loudly():
    with pytest.raises(ValidationError):
        DocumentChunk(**{**BASE, "effective_date": "not-a-date"})


def test_critique_flags_are_derived():
    r = CritiqueReport(unsupported_claims=[UnsupportedClaim(claim="a", reason="b")], is_clean=True)
    assert r.has_hallucination and not r.is_clean and not r.has_corpus_conflict
    assert CritiqueReport().is_clean


def test_resolution_consistency():
    with pytest.raises(ValidationError):
        ResolutionOutput(status="UNRESOLVABLE_CONFLICT", final_answer="x", winning_sources=["D1"], rationale="r")

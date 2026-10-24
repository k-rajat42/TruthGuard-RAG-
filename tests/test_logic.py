"""LLM-free tests for deterministic logic. Run: pytest tests/test_logic.py"""
import pytest

from src.agents import _analyze_metadata_graph, _decide, adjudicate
from src.graph import route_after_critic
from src.retriever import CorpusError, annotate_supersession, current_chunks, validate_chunks
from src.schemas import ContradictionItem, CritiqueReport, DocumentChunk, RetrievedChunk, UnsupportedClaim


def rc(doc_id, doc_type="policy", date="2025-01-01", supersedes=None, topic="t", idx=0):
    ch = DocumentChunk(text="x", doc_id=doc_id, doc_type=doc_type, effective_date=date, version="1",
                       supersedes=supersedes, topic=topic, chunk_index=idx)
    return RetrievedChunk(chunk=ch, rrf_score=0.1)


def conflict(a, b):
    return CritiqueReport(contradictions=[ContradictionItem(topic="t", doc_id_a=a, doc_id_b=b, position_a="p", position_b="q")])


def test_supersession_flag_and_stale_hiding():
    pool = annotate_supersession([rc("OLD", date="2023-01-01"), rc("NEW", "amendment", "2025-03-01", supersedes="OLD")])
    assert [f.kind for f in _analyze_metadata_graph(pool)] == ["supersedes_link"]
    assert [r.chunk.doc_id for r in current_chunks(pool)] == ["NEW"]


def test_supersedes_resolves_even_against_later_date_or_rank():
    pool = [rc("A", "faq", "2025-01-01"), rc("B", "memo", "2024-01-01", supersedes="A")]
    adj = adjudicate(pool, conflict("A", "B"))
    assert adj.status == "RESOLVED_BY_PROVENANCE" and adj.winners == ["B"] and adj.rejected == ["A"]


def test_authority_hierarchy_when_dates_equal():
    pool = [rc("POL", "policy"), rc("FAQ", "faq")]
    assert "authority_mismatch" in [f.kind for f in _analyze_metadata_graph(pool)]
    assert adjudicate(pool, conflict("POL", "FAQ")).winners == ["POL"]


def test_equal_everything_is_unresolvable():
    pool = [rc("M1", "memo", "2025-01-20"), rc("M2", "memo", "2025-01-20")]
    assert "equal_authority_deadlock" in [f.kind for f in _analyze_metadata_graph(pool)]
    adj = adjudicate(pool, conflict("M1", "M2"))
    assert adj.status == "UNRESOLVABLE_CONFLICT" and adj.winners == []


def test_clean_and_topicless():
    assert adjudicate([rc("A")], CritiqueReport()).status == "VERIFIED_CLEAN"
    assert _analyze_metadata_graph([rc("A", topic=None), rc("B", "memo", topic=None)]) == []


def test_routing():
    bad = CritiqueReport(unsupported_claims=[UnsupportedClaim(claim="c", reason="r")])
    assert route_after_critic({"critique": bad, "revision_count": 0}) == "synthesizer"
    assert route_after_critic({"critique": bad, "revision_count": 2}) == "resolver"
    assert route_after_critic({"critique": CritiqueReport(), "revision_count": 0}) == "resolver"
    assert route_after_critic({"critique": conflict("A", "B"), "revision_count": 0}) == "resolver"


def test_validate_chunks_fail_fast_and_indexing():
    base = {"text": "x", "doc_id": "D", "doc_type": "memo", "effective_date": "2025-01-01", "version": "1"}
    out = validate_chunks([base, base | {"text": "y"}])
    assert [c.chunk_index for c in out] == [0, 1]
    with pytest.raises(CorpusError):
        validate_chunks([base, base | {"effective_date": "nope"}])

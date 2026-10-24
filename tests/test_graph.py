from unittest.mock import patch
from src.agents import ResolutionDraft
from src.graph import build_graph, run_query
from src.retriever import HybridRetriever
from src.schemas import (
    Claim,
    ContradictionItem,
    CritiqueReport,
    SynthesizerOutput,
    UnsupportedClaim,
)


def test_graph_clean_flow():
    r = HybridRetriever(path=":memory:")
    r.ingest([
        {"text": "Employees receive 20 days of paid vacation per year.", "doc_id": "DOC-VAC", "doc_type": "policy", "effective_date": "2024-01-01", "version": "1.0"}
    ])
    graph = build_graph(r)

    draft_mock = SynthesizerOutput(
        answer="Employees receive 20 days of paid vacation per year [DOC-VAC].",
        claims=[Claim(statement="Employees receive 20 days of paid vacation per year [DOC-VAC].", citations=["DOC-VAC"])],
    )
    critique_clean = CritiqueReport(
        unsupported_claims=[],
        contradictions=[],
        stale_citations=[],
        summary="No issues found.",
    )

    with patch("src.agents.invoke_structured") as mock_invoke:
        mock_invoke.side_effect = [draft_mock, critique_clean]

        res = run_query(graph, "How many vacation days do employees get?")
        assert res["resolution"].status == "VERIFIED_CLEAN"
        assert "20 days" in res["resolution"].final_answer
        assert res["revision_count"] == 0


def test_graph_retry_on_hallucination():
    r = HybridRetriever(path=":memory:")
    r.ingest([
        {"text": "Remote stipend is $400 monthly.", "doc_id": "DOC-STP", "doc_type": "policy", "effective_date": "2024-01-01", "version": "1.0"}
    ])
    graph = build_graph(r)

    draft_1 = SynthesizerOutput(
        answer="Stipend is $1000 and lunch is free [DOC-STP].",
        claims=[Claim(statement="Stipend is $1000 and lunch is free [DOC-STP].", citations=["DOC-STP"])],
    )
    critique_1 = CritiqueReport(
        unsupported_claims=[UnsupportedClaim(claim="lunch is free", reason="Not stated in docs")],
    )
    draft_2 = SynthesizerOutput(
        answer="Remote stipend is $400 monthly [DOC-STP].",
        claims=[Claim(statement="Remote stipend is $400 monthly [DOC-STP].", citations=["DOC-STP"])],
    )
    critique_clean = CritiqueReport()

    with patch("src.agents.invoke_structured") as mock_invoke:
        mock_invoke.side_effect = [draft_1, critique_1, draft_2, critique_clean]

        res = run_query(graph, "What is the stipend?")
        assert res["resolution"].status == "VERIFIED_CLEAN"
        assert res["revision_count"] == 1
        assert len([t for t in res["trace"] if t["step"] == "synthesizer"]) == 2

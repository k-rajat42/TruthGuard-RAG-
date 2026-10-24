import json
import pytest
from pathlib import Path

from src.retriever import CorpusError, HybridRetriever, current_chunks, load_chunks_from_file, main, validate_chunks
from src.schemas import DocumentChunk


def test_load_chunks_json(tmp_path: Path):
    f = tmp_path / "test.json"
    data = [{"text": "Sample text", "doc_id": "D1", "doc_type": "policy", "effective_date": "2024-01-01", "version": "1.0"}]
    f.write_text(json.dumps(data))
    records = load_chunks_from_file(f)
    assert len(records) == 1
    assert records[0]["doc_id"] == "D1"


def test_load_chunks_jsonl(tmp_path: Path):
    f = tmp_path / "test.jsonl"
    line = json.dumps({"text": "Line text", "doc_id": "D2", "doc_type": "memo", "effective_date": "2024-02-01", "version": "1.0"})
    f.write_text(f"{line}\n")
    records = load_chunks_from_file(f)
    assert len(records) == 1
    assert records[0]["doc_id"] == "D2"


def test_load_invalid_file(tmp_path: Path):
    f = tmp_path / "bad.txt"
    f.write_text("not json")
    with pytest.raises(CorpusError):
        load_chunks_from_file(f)


def test_retriever_ingest_and_retrieve():
    r = HybridRetriever(path=":memory:")
    chunks = [
        {"text": "Remote stipend is 500 dollars per month.", "doc_id": "DOC-1", "doc_type": "policy", "effective_date": "2024-01-01", "version": "1.0"},
        {"text": "Remote stipend is 600 dollars per month.", "doc_id": "DOC-2", "doc_type": "amendment", "effective_date": "2024-06-01", "version": "2.0", "supersedes": ["DOC-1"]},
    ]
    assert r.ingest(chunks) == 2
    assert r.count() == 2

    # Query
    pool = r.retrieve("stipend", top_k=2)
    assert len(pool) == 2
    doc_ids = {p.chunk.doc_id for p in pool}
    assert "DOC-1" in doc_ids and "DOC-2" in doc_ids

    # Superseded annotation
    doc1 = next(p for p in pool if p.chunk.doc_id == "DOC-1")
    assert "DOC-2" in doc1.superseded_by
    assert doc1.is_superseded

    # Synthesizer view should hide DOC-1
    synth_view = current_chunks(pool)
    assert [p.chunk.doc_id for p in synth_view] == ["DOC-2"]


def test_cli_main(tmp_path: Path):
    f = tmp_path / "corpus.json"
    data = [{"text": "Company travel policy", "doc_id": "DOC-TRV", "doc_type": "policy", "effective_date": "2024-01-01", "version": "1.0"}]
    f.write_text(json.dumps(data))

    exit_code = main(["--corpus", str(f), "--query", "travel policy", "--expect", "DOC-TRV"])
    assert exit_code == 0

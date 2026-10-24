# TruthGuard

Multi-agent RAG verifier: hybrid Qdrant retrieval -> Synthesizer -> adversarial Critic -> deterministic Resolver.

## Setup (Python 3.12)
    pip install -r requirements.txt
    cp .env.example .env      # add GOOGLE_API_KEY

## Plug in a corpus
A `.json` array or `.jsonl` of objects with: text, doc_id, doc_type, effective_date (YYYY-MM-DD), version,
and optional supersedes, topic, chunk_index (extra keys are kept in the payload).

## Verify retrieval (no LLM needed)
    # Using the included sample corpus:
    python -m src.retriever --corpus sample_corpus.json --query "What is the remote work stipend?" --expect DOC-HR-02 DOC-HR-01

    # Or test with your custom corpus:
    python -m src.retriever --corpus your.jsonl --query "..." --expect DOC-A DOC-B
## Logic tests
    pytest
## UI
    streamlit run app.py

"""Central configuration. Everything env-overridable; nothing hardcoded elsewhere."""
import json
import os
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

# Higher number = higher authority. Unknown doc_types rank 0.
_DEFAULT_HIERARCHY = {"amendment": 4, "policy": 3, "memo": 2, "faq": 1}


def _load_hierarchy() -> dict[str, int]:
    raw = os.getenv("TRUTHGUARD_DOC_HIERARCHY")
    if not raw:
        return dict(_DEFAULT_HIERARCHY)
    parsed = json.loads(raw)
    return {str(k).strip().lower(): int(v) for k, v in parsed.items()}


DOC_TYPE_HIERARCHY: dict[str, int] = _load_hierarchy()


def authority_rank(doc_type: str) -> int:
    """Single source of truth for authority comparisons."""
    return DOC_TYPE_HIERARCHY.get(doc_type.strip().lower(), 0)


@dataclass(frozen=True)
class Settings:
    qdrant_path: str = os.getenv("TRUTHGUARD_QDRANT_PATH", ":memory:")
    collection_name: str = os.getenv("TRUTHGUARD_COLLECTION", "truthguard_chunks")
    dense_model: str = "BAAI/bge-small-en-v1.5"  # 384-dim, cosine
    sparse_model: str = "Qdrant/bm25"
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "sparse"
    top_k: int = int(os.getenv("TRUTHGUARD_TOP_K", "6"))
    # Candidates pulled by EACH prefetch (dense + sparse) before RRF fusion. Lower = cheaper.
    prefetch_limit: int = int(os.getenv("TRUTHGUARD_PREFETCH_LIMIT", "10"))
    ingest_batch_size: int = int(os.getenv("TRUTHGUARD_INGEST_BATCH", "16"))
    rrf_k: int = 60
    max_revisions: int = 2
    synth_model: str = os.getenv("TRUTHGUARD_SYNTH_MODEL", "gemini-2.5-flash")
    critic_model: str = os.getenv("TRUTHGUARD_CRITIC_MODEL", "gemini-2.5-pro")
    resolver_model: str = os.getenv("TRUTHGUARD_RESOLVER_MODEL", "gemini-2.5-pro")
    # Used when a model keeps failing on quota/availability errors. Set empty to disable.
    fallback_model: str = os.getenv("TRUTHGUARD_FALLBACK_MODEL", "gemini-2.5-flash")
    ollama_base_url: str = os.getenv("TRUTHGUARD_OLLAMA_URL", "http://localhost:11434")
    ollama_num_ctx: int = int(os.getenv("TRUTHGUARD_OLLAMA_NUM_CTX", "8192"))
    llm_retries: int = int(os.getenv("TRUTHGUARD_LLM_RETRIES", "3"))


def get_settings() -> Settings:
    load_dotenv(override=True)
    return Settings(
        qdrant_path=os.getenv("TRUTHGUARD_QDRANT_PATH", ":memory:"),
        collection_name=os.getenv("TRUTHGUARD_COLLECTION", "truthguard_chunks"),
        top_k=int(os.getenv("TRUTHGUARD_TOP_K", "6")),
        prefetch_limit=int(os.getenv("TRUTHGUARD_PREFETCH_LIMIT", "10")),
        ingest_batch_size=int(os.getenv("TRUTHGUARD_INGEST_BATCH", "16")),
        synth_model=os.getenv("TRUTHGUARD_SYNTH_MODEL", "gemini-2.5-flash"),
        critic_model=os.getenv("TRUTHGUARD_CRITIC_MODEL", "gemini-2.5-pro"),
        resolver_model=os.getenv("TRUTHGUARD_RESOLVER_MODEL", "gemini-2.5-pro"),
        fallback_model=os.getenv("TRUTHGUARD_FALLBACK_MODEL", ""),
        ollama_base_url=os.getenv("TRUTHGUARD_OLLAMA_URL", "http://localhost:11434"),
        ollama_num_ctx=int(os.getenv("TRUTHGUARD_OLLAMA_NUM_CTX", "8192")),
        llm_retries=int(os.getenv("TRUTHGUARD_LLM_RETRIES", "3")),
    )


SETTINGS = get_settings()
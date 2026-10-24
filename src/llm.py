"""LLM access (Gemini or local Ollama) with structured output, retry/backoff and optional model fallback.

Built for free-tier quotas: transient errors (429 / 503 / timeouts) are retried with
exponential backoff, then the call falls back to TRUTHGUARD_FALLBACK_MODEL if set.
Model strings starting with "ollama:" run on your local Ollama; everything else uses Gemini.
"""
import time
from typing import Optional, Type, TypeVar

from pydantic import BaseModel

from src.config import SETTINGS

T = TypeVar("T", bound=BaseModel)

_TRANSIENT = ("429", "503", "500", "resource_exhausted", "unavailable", "deadline", "timeout", "rate limit", "quota", "overloaded")

OLLAMA_PREFIX = "ollama:"


def get_llm(model: str, temperature: float = 0.0):
    """Model string decides the provider: "ollama:<tag>" -> local Ollama, anything else -> Gemini."""
    # Lazy imports so pure-logic modules/tests work without the LLM stack installed.
    if model.startswith(OLLAMA_PREFIX):
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model[len(OLLAMA_PREFIX):],
            temperature=temperature,
            base_url=SETTINGS.ollama_base_url,
            num_ctx=SETTINGS.ollama_num_ctx,  # Ollama's small default context would truncate our prompts
        )
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=model, temperature=temperature)  # key from GOOGLE_API_KEY


def _is_transient(e: Exception) -> bool:
    s = str(e).lower()
    return any(t in s for t in _TRANSIENT)


def invoke_structured(schema: Type[T], model: str, messages: list, fallback_model: Optional[str] = None) -> T:
    """Call `model` with json_schema structured output; return a validated `schema` instance."""
    fallback = SETTINGS.fallback_model if fallback_model is None else fallback_model
    if model.startswith(OLLAMA_PREFIX) and fallback_model is None:
        fallback = ""  # a local model must never silently fall back to a cloud API
    candidates = [model] + ([fallback] if fallback and fallback != model else [])
    last: Optional[Exception] = None
    for idx, m in enumerate(candidates):
        try:
            chain = get_llm(m).with_structured_output(schema, method="json_schema")
        except Exception as e:  # noqa: BLE001
            last = e
            if idx < len(candidates) - 1:
                continue
            raise

        for attempt in range(SETTINGS.llm_retries):
            try:
                out = chain.invoke(messages)
                if out is None:
                    raise ValueError("model returned no parsable structured output (transient)")
                return out if isinstance(out, schema) else schema.model_validate(out)
            except Exception as e:  # noqa: BLE001
                last = e
                # If non-transient error, don't keep retrying the exact same model
                if not (_is_transient(e) or isinstance(e, ValueError)):
                    break
                if attempt < SETTINGS.llm_retries - 1:
                    time.sleep(min(2 ** (attempt + 1), 30))
    raise RuntimeError(f"LLM call failed after retries/fallback: {last}") from last
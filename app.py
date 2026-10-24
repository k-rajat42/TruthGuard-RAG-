"""TruthGuard Adversarial Inspector (Streamlit). Run: streamlit run app.py"""
import html
import importlib
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Ensure fresh environment and module definitions on every Streamlit run
load_dotenv(override=True)

import src.config
import src.schemas
import src.agents
import src.graph

importlib.reload(src.config)
importlib.reload(src.schemas)
importlib.reload(src.agents)
importlib.reload(src.graph)

import pandas as pd
import streamlit as st

from src.config import DOC_TYPE_HIERARCHY, SETTINGS, get_settings
from src.graph import build_graph, run_query
from src.retriever import CorpusError, HybridRetriever, current_chunks

st.set_page_config(page_title="TruthGuard Inspector", page_icon="🛡️", layout="wide")

BADGES = {
    "VERIFIED_CLEAN": ("🟢", "#14532d", "#dcfce7"),
    "RESOLVED_BY_PROVENANCE": ("🟡", "#713f12", "#fef9c3"),
    "UNRESOLVABLE_CONFLICT": ("🔴", "#7f1d1d", "#fee2e2"),
}


@st.cache_resource
def get_retriever() -> HybridRetriever:
    return HybridRetriever()


def get_graph(retriever: HybridRetriever):
    return build_graph(retriever)


def pill(text: str, bg: str) -> str:
    return (f'<span style="background:{bg};color:#fff;padding:3px 12px;border-radius:999px;'
            f'margin:2px 4px 2px 0;display:inline-block;font-size:0.85em">{html.escape(text)}</span>')


retriever = get_retriever()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🛡️ TruthGuard")
    cfg = get_settings()

    # Reset any stale llama reference in widget session state
    for role, key in [
        ("synth_model", "synth_model_input"),
        ("critic_model", "critic_model_input"),
        ("resolver_model", "resolver_model_input"),
    ]:
        val = getattr(cfg, role)
        if key not in st.session_state or "llama" in str(st.session_state.get(key, "")).lower():
            st.session_state[key] = val

    with st.expander("🤖 Model Configuration", expanded=False):
        synth_model = st.text_input("Synthesizer model", key="synth_model_input")
        critic_model = st.text_input("Critic model", key="critic_model_input")
        resolver_model = st.text_input("Resolver model", key="resolver_model_input")

    _roles = [synth_model, critic_model, resolver_model]
    st.caption("Active: " + " · ".join(f"{n} `{m}`" for n, m in zip(("synth", "critic", "resolver"), _roles)))

    if all(m.startswith("ollama:") for m in _roles):
        st.info(f"🖥️ Local Ollama mode active (`{cfg.ollama_base_url}`)")
    else:
        current_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
        api_key_input = st.text_input(
            "Google API Key",
            value=current_key,
            type="password",
            help="Required for Gemini models. Enter here or set in .env",
        )
        if api_key_input:
            os.environ["GOOGLE_API_KEY"] = api_key_input
        elif not current_key:
            st.warning("Gemini models require a GOOGLE_API_KEY. Enter your key above or switch roles to ollama: models.")

    st.subheader("1 · Corpus")
    up = st.file_uploader("JSON array or JSONL of DocumentChunk records", type=["json", "jsonl"])
    wipe = st.checkbox("Replace existing index", value=True)
    if st.button("Load uploaded corpus", disabled=up is None, use_container_width=True):
        suffix = Path(up.name).suffix
        with tempfile.NamedTemporaryFile("wb", suffix=suffix, delete=False) as f:
            f.write(up.getvalue())
        try:
            if wipe:
                retriever.reset()
            n = retriever.ingest_file(f.name)
            st.success(f"Indexed {n} chunks.")
            st.rerun()
        except CorpusError as e:
            st.error(str(e))
        finally:
            os.unlink(f.name)

    sample_file = Path("sample_corpus.json")
    if sample_file.exists():
        if st.button("📂 Load demo corpus (8 chunks)", use_container_width=True):
            if wipe:
                retriever.reset()
            n = retriever.ingest_file(sample_file)
            st.success(f"Indexed {n} demo chunks!")
            st.rerun()

    st.subheader("2 · Retrieval")
    top_k = st.number_input("top_k", 1, 20, cfg.top_k)
    prefetch = st.number_input("prefetch_limit (per vector)", 1, 100, cfg.prefetch_limit)
    expand = st.checkbox("Expand supersession links", value=True)
    st.caption(f"Authority hierarchy: {DOC_TYPE_HIERARCHY}")

    with st.expander(f"Qdrant payload explorer ({retriever.count()} points)"):
        pts = retriever.list_points()
        if pts:
            df = pd.DataFrame(pts)
            df["text"] = df["text"].str.slice(0, 80)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("Index is empty.")

# ------------------------------------------------------------------ main
st.header("Adversarial Inspector")

if "query_input" not in st.session_state:
    st.session_state["query_input"] = ""

st.caption("Try an example query:")
c1, c2, c3, c4 = st.columns(4)
if c1.button("💰 Remote stipend", use_container_width=True, help="Tests supersession resolution"):
    st.session_state["query_input"] = "What is the monthly remote work stipend?"
if c2.button("🔒 Laptop security", use_container_width=True, help="Tests authority hierarchy (Policy vs Memo)"):
    st.session_state["query_input"] = "What are the laptop encryption requirements?"
if c3.button("✈️ Meal per diem", use_container_width=True, help="Tests unresolvable conflict (equal authority deadlock)"):
    st.session_state["query_input"] = "What is the daily meal per diem for domestic travel?"
if c4.button("🧾 Expense receipt", use_container_width=True, help="Tests clean verification"):
    st.session_state["query_input"] = "When are itemized receipts required for expenses?"

query = st.text_area(
    "Ask a question about your corpus",
    value=st.session_state["query_input"],
    height=80,
    placeholder="e.g. What is the remote work stipend?",
)
if st.button("Run TruthGuard", type="primary", disabled=not query.strip() or retriever.count() == 0):
    with st.spinner("Retrieving → synthesizing → critiquing → resolving..."):
        try:
            st.session_state["result"] = run_query(
                get_graph(retriever),
                query.strip(),
                top_k=int(top_k),
                prefetch_limit=int(prefetch),
                expand_supersession=expand,
                synth_model=synth_model,
                critic_model=critic_model,
                resolver_model=resolver_model,
            )
        except Exception as e:  # noqa: BLE001
            st.session_state.pop("result", None)
            st.error(f"Pipeline failed: {e}")
if retriever.count() == 0:
    st.info("Load a corpus in the sidebar to begin.")

res = st.session_state.get("result")
if res:
    out = res["resolution"]
    icon, fg, bg = BADGES[out.status]
    st.markdown(
        f'<div style="background:{bg};color:{fg};padding:14px 18px;border-radius:10px;'
        f'font-size:1.25em;font-weight:700">{icon} {out.status}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("#### Final answer")
    st.write(out.final_answer)
    st.caption(out.rationale)

    if out.status != "VERIFIED_CLEAN":
        st.markdown("#### Provenance")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Winning sources**")
            st.markdown("".join(pill(d, "#15803d") for d in out.winning_sources) or "_none (no winner can be named)_",
                        unsafe_allow_html=True)
        with c2:
            st.markdown("**Superseded / rejected sources**")
            st.markdown("".join(pill(d, "#b91c1c") for d in out.rejected_sources) or "_none_", unsafe_allow_html=True)
        for p in out.positions:
            st.markdown(f"- **{p.doc_id}**: {p.claim}")
        for r in out.rules_applied:
            st.caption(f"Rule: {r}")
    if out.dropped_unsupported_claims:
        st.warning("Removed after max retries: " + "; ".join(out.dropped_unsupported_claims))

    # -------------------------------------------------------------- 3-column trace
    st.markdown("---")
    st.markdown("#### Pipeline trace")
    col1, col2, col3 = st.columns(3)
    pool = res["retrieved"]
    synth_ids = {r.chunk.doc_id for r in current_chunks(pool)}

    with col1:
        st.markdown("##### ① Qdrant hybrid pool")
        st.caption("Dense + BM25 fused by RRF. Stale chunks are kept here but hidden from the Synthesizer.")
        st.dataframe(pd.DataFrame([{
            "doc_id": r.chunk.doc_id, "type": r.chunk.doc_type, "date": str(r.chunk.effective_date),
            "ver": r.chunk.version, "topic": r.chunk.topic, "rrf": round(r.rrf_score, 4),
            "src": r.retrieval_source, "superseded_by": ", ".join(r.superseded_by),
            "to synth": r.chunk.doc_id in synth_ids,
        } for r in pool]), hide_index=True, use_container_width=True)
        for r in pool:
            with st.expander(f"{r.chunk.doc_id} · chunk {r.chunk.chunk_index}"):
                st.write(r.chunk.text)

    with col2:
        st.markdown("##### ② Synthesizer draft(s)")
        for e in (t for t in res["trace"] if t["step"] == "synthesizer"):
            st.markdown(f"**Attempt {e['attempt'] + 1}**")
            st.info(e["draft"]["answer"])
            for c in e["draft"]["claims"]:
                st.markdown(f"- {c['statement']}  \n  _cites: {', '.join(c['citations'])}_")

    with col3:
        st.markdown("##### ③ Adversarial critic")
        flags = res.get("metadata_flags", [])
        with st.expander(f"Deterministic metadata flags ({len(flags)})", expanded=bool(flags)):
            for f in flags:
                st.markdown(f"- `{f.kind}` {f.detail}")
        for e in (t for t in res["trace"] if t["step"] == "critic"):
            c = e["critique"]
            st.markdown(f"**Verdict on attempt {e['attempt'] + 1}**")
            st.markdown(
                pill("hallucination" if c["has_hallucination"] else "no hallucination", "#b91c1c" if c["has_hallucination"] else "#15803d")
                + pill("corpus conflict" if c["has_corpus_conflict"] else "no corpus conflict", "#b45309" if c["has_corpus_conflict"] else "#15803d"),
                unsafe_allow_html=True,
            )
            for u in c["unsupported_claims"]:
                st.error(f"Unsupported: {u['claim']}  \n_{u['reason']}_")
            for x in c["contradictions"]:
                st.warning(f"Contradiction on {x['topic']}: **{x['doc_id_a']}** says {x['position_a']} vs **{x['doc_id_b']}** says {x['position_b']}")
            for s in c["stale_citations"]:
                st.warning(f"Stale: {s['stale_doc_id']} superseded by {s['superseding_doc_id']} ({s['claim']})")
            if c["summary"]:
                st.caption(c["summary"])
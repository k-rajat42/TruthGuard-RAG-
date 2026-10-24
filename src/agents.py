"""TruthGuard agents.

Part 1: synthesizer_node, _analyze_metadata_graph (deterministic), critic_node.
Part 2: adjudicate (deterministic rules) + resolution_node.

Design principle: the LLM judges CONTENT (does the draft match the sources? do two
documents actually disagree?). Python decides AUTHORITY (supersedes > date > doc_type).
"""
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.config import SETTINGS, authority_rank
from src.llm import invoke_structured
from src.retriever import current_chunks
from src.schemas import (
    CritiqueReport,
    DocumentChunk,
    MetadataFlag,
    ResolutionOutput,
    RetrievedChunk,
    SourcePosition,
    StaleCitation,
    SynthesizerOutput,
    TruthGuardState,
    UnsupportedClaim,
)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _fmt_chunk(r: RetrievedChunk) -> str:
    c = r.chunk
    topic = f" | topic={c.topic}" if c.topic else ""
    stale = f" | SUPERSEDED_BY={','.join(r.superseded_by)}" if r.superseded_by else ""
    sup = f" | supersedes={','.join(c.supersedes)}" if c.supersedes else ""
    return f"[{c.doc_id}] type={c.doc_type} | effective={c.effective_date} | version={c.version}{topic}{sup}{stale}\n{c.text}"


def _fmt_pool(pool: list[RetrievedChunk]) -> str:
    return "\n\n---\n\n".join(_fmt_chunk(r) for r in pool)


def _doc_index(pool: list[RetrievedChunk]) -> dict[str, DocumentChunk]:
    idx: dict[str, DocumentChunk] = {}
    for r in pool:
        idx.setdefault(r.chunk.doc_id, r.chunk)
    return idx


# --------------------------------------------------------------------------- #
# Synthesizer
# --------------------------------------------------------------------------- #
SYNTH_SYSTEM = """You are the Synthesizer in a fact-verification pipeline. Answer the question using ONLY the source chunks provided.

Rules:
1. Every sentence of `answer` must end with an inline citation such as [DOC-ID] naming the chunk(s) that state it. Cite only doc_ids shown in the sources.
2. Never add facts, numbers, dates or inferences that are not stated in the sources.
3. If sources disagree with each other, state each position with its own citation. Do NOT pick a winner, average, or blend them; resolving conflicts is not your job.
4. If the sources do not answer the question, set `answer` to "The provided documents do not contain this information." and `claims` to [].
5. `claims` must restate the answer one atomic sentence per item, each with the doc_ids that support it."""


def synthesizer_node(state: TruthGuardState) -> dict:
    pool = state["retrieved"]
    view = current_chunks(pool)  # stale (explicitly superseded) chunks are hidden from the Synthesizer
    feedback = state.get("critique_feedback")
    is_retry = bool(feedback)
    revision_count = state.get("revision_count", 0) + (1 if is_retry else 0)

    user = f"QUESTION:\n{state['query']}\n\nSOURCES:\n{_fmt_pool(view)}"
    if is_retry and state.get("draft"):
        user += (
            "\n\nYour previous draft was REJECTED by an adversarial critic.\n"
            f"Previous draft:\n{state['draft'].answer}\n\n"
            f"Problems to fix (remove or correct each, using only the sources):\n{feedback}\n"
            "Write a corrected draft."
        )
    synth_model = state.get("synth_model") or SETTINGS.synth_model
    draft = invoke_structured(
        SynthesizerOutput, synth_model, [SystemMessage(content=SYNTH_SYSTEM), HumanMessage(content=user)]
    )
    entry = {"step": "synthesizer", "attempt": revision_count, "draft": draft.model_dump(mode="json")}
    return {"draft": draft, "revision_count": revision_count, "trace": state.get("trace", []) + [entry]}


# --------------------------------------------------------------------------- #
# Deterministic pre-critic metadata analysis
# --------------------------------------------------------------------------- #
def _analyze_metadata_graph(pool: list[RetrievedChunk]) -> list[MetadataFlag]:
    """Pure-Python inspection of retrieved payloads. No LLM.

    Flags (1) explicit supersedes links between retrieved docs and (2) date / authority
    mismatches between docs sharing a `topic`. Chunks without a topic are skipped for (2).
    """
    flags: list[MetadataFlag] = []
    docs = _doc_index(pool)
    linked: set[frozenset[str]] = set()

    for d in sorted(docs.values(), key=lambda x: x.doc_id):
        for old in d.supersedes:
            if old in docs and old != d.doc_id:
                linked.add(frozenset((d.doc_id, old)))
                flags.append(
                    MetadataFlag(
                        kind="supersedes_link",
                        doc_ids=[d.doc_id, old],
                        detail=f"{d.doc_id} (effective {d.effective_date}) explicitly supersedes {old} (effective {docs[old].effective_date}).",
                    )
                )

    by_topic: dict[str, dict[str, DocumentChunk]] = defaultdict(dict)
    for r in pool:
        if r.chunk.topic:
            by_topic[r.chunk.topic].setdefault(r.chunk.doc_id, r.chunk)

    for topic, group in sorted(by_topic.items()):
        for a, b in combinations(sorted(group.values(), key=lambda x: x.doc_id), 2):
            if frozenset((a.doc_id, b.doc_id)) in linked:
                continue
            ra, rb = authority_rank(a.doc_type), authority_rank(b.doc_type)
            ids = [a.doc_id, b.doc_id]
            if a.effective_date != b.effective_date:
                flags.append(MetadataFlag(kind="date_mismatch", doc_ids=ids, topic=topic,
                    detail=f"Topic '{topic}': {a.doc_id} is dated {a.effective_date}, {b.doc_id} is dated {b.effective_date}."))
            if ra != rb:
                flags.append(MetadataFlag(kind="authority_mismatch", doc_ids=ids, topic=topic,
                    detail=f"Topic '{topic}': {a.doc_id} is a {a.doc_type} (rank {ra}), {b.doc_id} is a {b.doc_type} (rank {rb})."))
            if a.effective_date == b.effective_date and ra == rb:
                flags.append(MetadataFlag(kind="equal_authority_deadlock", doc_ids=ids, topic=topic,
                    detail=f"Topic '{topic}': {a.doc_id} and {b.doc_id} share date {a.effective_date} and authority rank {ra}, with no supersedes link."))
    return flags


def _deterministic_citation_checks(
    draft: SynthesizerOutput, pool: list[RetrievedChunk]
) -> tuple[list[UnsupportedClaim], list[StaleCitation]]:
    """Cheap checks that need no LLM: citations to docs that were never retrieved, or that are stale."""
    ids = {r.chunk.doc_id for r in pool}
    superseded = {r.chunk.doc_id: r.superseded_by for r in pool if r.superseded_by}
    unsupported: list[UnsupportedClaim] = []
    stale: list[StaleCitation] = []
    for c in draft.claims:
        unknown = [d for d in c.citations if d not in ids]
        if unknown:
            unsupported.append(UnsupportedClaim(claim=c.statement, reason=f"Cites doc_id(s) {unknown} that were never retrieved."))
        for d in c.citations:
            if d in superseded:
                stale.append(StaleCitation(claim=c.statement, stale_doc_id=d, superseding_doc_id=superseded[d][0]))
    return unsupported, stale


# --------------------------------------------------------------------------- #
# Critic
# --------------------------------------------------------------------------- #
CRITIC_SYSTEM = """You are an adversarial Critic. You audit a draft answer against the retrieved source documents.
There are two DIFFERENT kinds of problem. Never confuse them:

* CORPUS CONFLICT = two retrieved DOCUMENTS disagree with each other on a fact relevant to the question (or the draft relies on a document another retrieved document supersedes). This is the documents' fault, not the draft's.
* HALLUCINATION = the DRAFT asserts something no retrieved chunk states (invented or altered number, date, name, policy, or a citation that does not support the sentence).

Run three checks:
1. UNSUPPORTED CLAIMS: for each draft claim, verify the cited chunk actually states it, with exact numbers. List failures in `unsupported_claims`. A disagreement between documents is NOT a hallucination. A statement that the documents lack the information is not a hallucination.
2. CONTRADICTIONS: list pairs of retrieved documents that give different answers to the same question-relevant fact, quoting each position in `position_a` / `position_b` from the chunk text. Do not list pairs that agree. Use the METADATA FLAGS as hints about where to look; a flag alone is not a contradiction.
3. STALENESS: if the draft relies on a document marked SUPERSEDED_BY (or otherwise outdated by a newer retrieved document), list it in `stale_citations`.

Leave `has_hallucination`, `has_corpus_conflict` and `is_clean` as false; they are computed for you. Use empty lists when a check finds nothing. Be strict but never invent problems."""


def critic_node(state: TruthGuardState) -> dict:
    pool = state["retrieved"]
    draft = state["draft"]
    flags = _analyze_metadata_graph(pool)
    ids = {r.chunk.doc_id for r in pool}

    flag_text = "\n".join(f"- {f.kind}: {f.detail}" for f in flags) or "(none)"
    claims_text = "\n".join(f"- {c.statement}  (cites: {', '.join(c.citations)})" for c in draft.claims) or "(no claims)"
    user = (
        f"QUESTION:\n{state['query']}\n\n"
        f"RETRIEVED CHUNKS (full pool, including stale ones):\n{_fmt_pool(pool)}\n\n"
        f"DRAFT ANSWER:\n{draft.answer}\n\nDRAFT CLAIMS:\n{claims_text}\n\n"
        f"METADATA FLAGS (deterministic):\n{flag_text}"
    )
    critic_model = state.get("critic_model") or SETTINGS.critic_model
    llm = invoke_structured(
        CritiqueReport, critic_model, [SystemMessage(content=CRITIC_SYSTEM), HumanMessage(content=user)]
    )

    # Guard against invented doc_ids, then merge deterministic findings.
    contradictions = [c for c in llm.contradictions if c.doc_id_a in ids and c.doc_id_b in ids and c.doc_id_a != c.doc_id_b]
    stale = [s for s in llm.stale_citations if s.stale_doc_id in ids and s.superseding_doc_id in ids]
    unsupported = list(llm.unsupported_claims)

    det_unsup, det_stale = _deterministic_citation_checks(draft, pool)
    seen_u = {u.claim.strip().lower() for u in unsupported}
    unsupported += [u for u in det_unsup if u.claim.strip().lower() not in seen_u]
    seen_s = {(s.stale_doc_id, s.claim.strip().lower()) for s in stale}
    stale += [s for s in det_stale if (s.stale_doc_id, s.claim.strip().lower()) not in seen_s]

    report = CritiqueReport(  # validator recomputes the booleans deterministically
        contradictions=contradictions, stale_citations=stale, unsupported_claims=unsupported, summary=llm.summary
    )
    feedback = None
    if report.has_hallucination:
        feedback = "\n".join(f"- \"{u.claim}\" -> {u.reason}" for u in report.unsupported_claims)

    entry = {
        "step": "critic",
        "attempt": state.get("revision_count", 0),
        "critique": report.model_dump(mode="json"),
        "metadata_flags": [f.model_dump(mode="json") for f in flags],
    }
    return {"critique": report, "metadata_flags": flags, "critique_feedback": feedback,
            "trace": state.get("trace", []) + [entry]}


# --------------------------------------------------------------------------- #
# Resolution: deterministic adjudication + LLM narration
# --------------------------------------------------------------------------- #
@dataclass
class Adjudication:
    status: str
    winners: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)
    deadlocks: list[str] = field(default_factory=list)
    pairs: list[tuple[str, str]] = field(default_factory=list)


def _decide(a: DocumentChunk, b: DocumentChunk) -> tuple[Optional[str], Optional[str], str]:
    """Return (winner, loser, rule). Order: supersedes link > effective_date > doc_type hierarchy."""
    if a.doc_id in b.supersedes:
        return b.doc_id, a.doc_id, "explicit supersedes link"
    if b.doc_id in a.supersedes:
        return a.doc_id, b.doc_id, "explicit supersedes link"
    if a.effective_date != b.effective_date:
        w, l = (a, b) if a.effective_date > b.effective_date else (b, a)
        return w.doc_id, l.doc_id, f"later effective_date ({w.effective_date} > {l.effective_date})"
    ra, rb = authority_rank(a.doc_type), authority_rank(b.doc_type)
    if ra != rb:
        w, l = (a, b) if ra > rb else (b, a)
        return w.doc_id, l.doc_id, f"higher authority ({w.doc_type} > {l.doc_type})"
    return None, None, "equal effective_date and authority, no supersedes link"


def adjudicate(pool: list[RetrievedChunk], critique: CritiqueReport) -> Adjudication:
    docs = _doc_index(pool)
    raw = [(c.doc_id_a, c.doc_id_b) for c in critique.contradictions]
    raw += [(s.superseding_doc_id, s.stale_doc_id) for s in critique.stale_citations]
    pairs, seen = [], set()
    for a, b in raw:
        k = frozenset((a, b))
        if k not in seen and a in docs and b in docs:
            seen.add(k)
            pairs.append((a, b))
    if not pairs:
        return Adjudication(status="VERIFIED_CLEAN")

    winners: set[str] = set()
    rejected: set[str] = set()
    adj = Adjudication(status="", pairs=pairs)
    for a, b in pairs:
        w, l, rule = _decide(docs[a], docs[b])
        if w is None:
            adj.deadlocks.append(f"{a} vs {b}: {rule}")
        else:
            winners.add(w)
            rejected.add(l)
            adj.rules.append(f"{w} over {l}: {rule}")
    winners -= rejected
    if adj.deadlocks or not winners:
        adj.status, adj.winners = "UNRESOLVABLE_CONFLICT", []
    else:
        adj.status, adj.winners = "RESOLVED_BY_PROVENANCE", sorted(winners)
    adj.rejected = sorted(rejected)
    return adj


class ResolutionDraft(BaseModel):
    """LLM-facing schema: only the prose. Status/winners are decided by Python, not the LLM."""

    final_answer: str = Field(description="User-facing answer; every sentence ends with an inline [DOC-ID] citation.")
    rationale: str = Field(description="Two or three sentences explaining the verdict and the rule applied.")


RESOLVER_SYSTEM = """You write the final user-facing answer for a fact-verification pipeline. The VERDICT was already decided by deterministic rules; you must follow it exactly and never override it.
Rules for every status:
- Use ONLY facts stated in the sources. Every sentence ends with an inline [DOC-ID] citation.
- Never blend or average conflicting values.
Per status:
- RESOLVED_BY_PROVENANCE: give the answer from the WINNING source(s). Then state the superseded/outranked position(s) with citation and explain in one sentence why the winner prevails, naming the rule.
- UNRESOLVABLE_CONFLICT: do NOT choose a side. State plainly that the documents conflict and cannot be reconciled, present each position with its citation, and suggest confirming with the document owner.
- VERIFIED_CLEAN: answer concisely from the sources, omitting any claim the sources do not support."""


def resolution_node(state: TruthGuardState) -> dict:
    pool, draft, critique = state["retrieved"], state["draft"], state["critique"]
    adj = adjudicate(pool, critique)
    docs = _doc_index(pool)
    dropped = [u.claim for u in critique.unsupported_claims]

    # Positions: critic's quoted positions when available, else a text snippet.
    positions: list[SourcePosition] = []
    for c in critique.contradictions:
        positions += [SourcePosition(doc_id=c.doc_id_a, claim=f"{c.position_a} [{c.doc_id_a}]"),
                      SourcePosition(doc_id=c.doc_id_b, claim=f"{c.position_b} [{c.doc_id_b}]")]
    have = {p.doc_id for p in positions}
    for a, b in adj.pairs:
        for d in (a, b):
            if d not in have:
                have.add(d)
                positions.append(SourcePosition(doc_id=d, claim=f"{docs[d].text[:240]} [{d}]"))

    if adj.status == "VERIFIED_CLEAN" and not dropped:
        # Nothing to adjudicate and the draft is verified: no LLM call needed (saves quota).
        out = ResolutionOutput(
            status="VERIFIED_CLEAN", final_answer=draft.answer,
            rationale="No contradictions, stale citations or unsupported claims were found; the verified draft is returned unchanged.",
        )
    else:
        verdict = (
            f"VERDICT: {adj.status}\nWINNING: {adj.winners}\nREJECTED: {adj.rejected}\n"
            f"RULES: {adj.rules or 'n/a'}\nDEADLOCKS: {adj.deadlocks or 'none'}\n"
            f"UNSUPPORTED CLAIMS TO OMIT: {dropped or 'none'}"
        )
        user = f"QUESTION:\n{state['query']}\n\n{verdict}\n\nSOURCES:\n{_fmt_pool(pool)}"
        resolver_model = state.get("resolver_model") or SETTINGS.resolver_model
        narr = invoke_structured(
            ResolutionDraft, resolver_model, [SystemMessage(content=RESOLVER_SYSTEM), HumanMessage(content=user)]
        )
        rationale = narr.rationale
        if dropped:
            rationale += f" ({len(dropped)} claim(s) remained unsupported after {state.get('revision_count', 0)} retries and were removed.)"
        out = ResolutionOutput(
            status=adj.status, final_answer=narr.final_answer,
            winning_sources=adj.winners, rejected_sources=adj.rejected,
            positions=positions if adj.status != "VERIFIED_CLEAN" else [],
            rules_applied=adj.rules + adj.deadlocks, dropped_unsupported_claims=dropped, rationale=rationale,
        )
    entry = {"step": "resolver", "resolution": out.model_dump(mode="json")}
    return {"resolution": out, "trace": state.get("trace", []) + [entry]}

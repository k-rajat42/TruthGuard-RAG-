"""LangGraph wiring.

retrieve -> synthesizer -> critic
critic -> synthesizer   if has_hallucination and revision_count < max_revisions
critic -> resolver      otherwise (clean, corpus conflict, or retries exhausted)
resolver -> END
"""
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agents import critic_node, resolution_node, synthesizer_node
from src.config import SETTINGS
from src.retriever import HybridRetriever
from src.schemas import TruthGuardState


def route_after_critic(state: TruthGuardState) -> str:
    c = state["critique"]
    if c.has_hallucination and state.get("revision_count", 0) < SETTINGS.max_revisions:
        return "synthesizer"
    return "resolver"  # clean, corpus conflict, or retries exhausted


def build_graph(retriever: HybridRetriever):
    def retrieve_node(state: TruthGuardState) -> dict:
        pool = retriever.retrieve(
            state["query"],
            top_k=state.get("top_k"),
            prefetch_limit=state.get("prefetch_limit"),
            expand_supersession=state.get("expand_supersession", True),
        )
        return {"retrieved": pool, "revision_count": 0, "critique_feedback": None, "trace": []}

    g = StateGraph(TruthGuardState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("synthesizer", synthesizer_node)
    g.add_node("critic", critic_node)
    g.add_node("resolver", resolution_node)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "synthesizer")
    g.add_edge("synthesizer", "critic")
    g.add_conditional_edges("critic", route_after_critic, {"synthesizer": "synthesizer", "resolver": "resolver"})
    g.add_edge("resolver", END)
    return g.compile()


def run_query(graph, query: str, **opts: Any) -> TruthGuardState:
    """opts: top_k, prefetch_limit, expand_supersession"""
    return graph.invoke({"query": query, **{k: v for k, v in opts.items() if v is not None}})

"""Compile the UDA-Hub LangGraph application.

Topology (Supervisor pattern):

    START -> hydrate -> classifier -> retriever -> [supervisor_router]
                                                        |
                                  +---------------------+----------------------+
                                  |                                            |
                              resolver                                    escalation
                                  |                                            |
                                  +-----------------> memory_writer -----------+
                                                            |
                                                           END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from uda_hub.agents.classifier import classifier_node
from uda_hub.agents.escalation import escalation_node
from uda_hub.agents.resolver import resolver_node
from uda_hub.agents.retriever import retriever_node
from uda_hub.agents.state import AgentState
from uda_hub.agents.supervisor import (
    hydrate_node,
    memory_writer_node,
    supervisor_router,
)
from uda_hub.memory import get_checkpointer


def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("hydrate", hydrate_node)
    g.add_node("classifier", classifier_node)
    g.add_node("retriever", retriever_node)
    g.add_node("resolver", resolver_node)
    g.add_node("escalation", escalation_node)
    g.add_node("memory_writer", memory_writer_node)

    g.add_edge(START, "hydrate")
    g.add_edge("hydrate", "classifier")
    g.add_edge("classifier", "retriever")

    g.add_conditional_edges(
        "retriever",
        supervisor_router,
        {"resolver": "resolver", "escalation": "escalation"},
    )
    g.add_edge("resolver", "memory_writer")
    g.add_edge("escalation", "memory_writer")
    g.add_edge("memory_writer", END)
    return g


def build_app(*, with_checkpointer: bool = True):
    graph = build_graph()
    if with_checkpointer:
        return graph.compile(checkpointer=get_checkpointer())
    return graph.compile()

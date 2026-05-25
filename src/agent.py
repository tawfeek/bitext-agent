"""LangGraph wiring for the data analyst agent.

Graph shape:

    START -> router -> { out_of_scope: decline -> END
                       | structured / unstructured: agent <-> tools (loop)
                                                    agent -> END (no tool calls or max iters)
                       }

We build the ReAct loop manually (rather than using ``create_react_agent``)
so we have first-class access to iteration counts (for the max-iterations
fallback) and to the routing classification.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .config import SETTINGS
from .llm import build_chat_model
from .router import RouteDecision, build_router
from .tools import ALL_TOOLS


class AgentState(TypedDict):
    """Mutable state passed between graph nodes."""

    messages: Annotated[list[BaseMessage], add_messages]
    classification: Optional[str]
    route_reason: Optional[str]
    iterations: int


_AGENT_SYSTEM_PROMPT = """You are a data analyst for the Bitext Customer
Service Tagged Training Dataset.

Available columns: category, intent, instruction (customer message),
response (agent reply).

You have tools that let you list categories/intents, count rows, fetch
examples, search by free-text phrase, and compute intent distributions.

Guidelines:
- For "how many ..." questions, prefer count_records (chain it with
  list_intents / list_categories first if you need to discover the exact
  filter value).
- For "show me examples ..." questions, use get_examples.
- For paraphrased asks ("people wanting their money back", "complaints
  about late delivery"), use search_examples on the customer instruction.
- For "summarize ..." or "how do agents respond ..." questions, call
  get_examples with a larger n (e.g. 15-20) on the relevant filter, then
  write a concise summary of the *responses* (or *instructions*, depending
  on the question) drawing on those rows. Do not invent details that are
  not present in the rows you fetched.
- Never answer questions that are not about this dataset from general
  knowledge.

When you have enough information, produce a final answer message with no
tool calls.
"""


def _make_router_node():
    router = build_router()

    def router_node(state: AgentState) -> dict:
        last_human = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        if last_human is None:
            return {"classification": "out_of_scope", "route_reason": "no user message"}
        decision: RouteDecision = router.invoke({"question": last_human.content})
        return {
            "classification": decision.kind,
            "route_reason": decision.reason,
            "iterations": 0,
        }

    return router_node


def _decline_node(state: AgentState) -> dict:
    msg = (
        "That question is outside the scope of this agent — I can only answer "
        "questions about the Bitext customer-service dataset (categories, "
        "intents, example messages, and summaries of customer/agent text). "
        "Try asking, for example: 'What categories exist?' or 'Summarize the "
        "FEEDBACK category.'"
    )
    return {"messages": [AIMessage(content=msg)]}


def _make_agent_node():
    model = build_chat_model(temperature=0.0).bind_tools(ALL_TOOLS)

    def agent_node(state: AgentState) -> dict:
        # Inject the system prompt only on the first agent turn.
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=_AGENT_SYSTEM_PROMPT), *messages]
        response = model.invoke(messages)
        return {
            "messages": [response],
            "iterations": state.get("iterations", 0) + 1,
        }

    return agent_node


def _max_iterations_node(state: AgentState) -> dict:
    msg = (
        f"I wasn't able to reach a confident answer within "
        f"{SETTINGS.max_iterations} reasoning steps. "
        "Could you rephrase or narrow your question? "
        "(e.g. specify a category or intent name.)"
    )
    return {"messages": [AIMessage(content=msg)]}


def _route_after_router(
    state: AgentState,
) -> Literal["agent", "decline"]:
    return "decline" if state.get("classification") == "out_of_scope" else "agent"


def _route_after_agent(
    state: AgentState,
) -> Literal["tools", "max_iterations", "__end__"]:
    last = state["messages"][-1]
    has_tool_calls = (
        isinstance(last, AIMessage) and bool(getattr(last, "tool_calls", []))
    )
    if not has_tool_calls:
        return END
    if state.get("iterations", 0) >= SETTINGS.max_iterations:
        return "max_iterations"
    return "tools"


def build_graph(checkpointer=None):
    """Compile and return the agent graph.

    Args:
        checkpointer: Optional LangGraph checkpointer for persistence
            (used in Task 2). Pass ``None`` for a stateless run.
    """
    builder: StateGraph = StateGraph(AgentState)
    builder.add_node("router", _make_router_node())
    builder.add_node("decline", _decline_node)
    builder.add_node("agent", _make_agent_node())
    builder.add_node("tools", ToolNode(ALL_TOOLS))
    builder.add_node("max_iterations", _max_iterations_node)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        _route_after_router,
        {"agent": "agent", "decline": "decline"},
    )
    builder.add_edge("decline", END)
    builder.add_conditional_edges(
        "agent",
        _route_after_agent,
        {"tools": "tools", "max_iterations": "max_iterations", END: END},
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("max_iterations", END)

    return builder.compile(checkpointer=checkpointer)


__all__ = ["AgentState", "build_graph"]

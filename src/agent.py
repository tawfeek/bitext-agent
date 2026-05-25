"""LangGraph wiring for the data analyst agent.

Graph shape (Task 2):

    START -> router -> { out_of_scope: decline -> END
                       | structured / unstructured: agent <-> tools (loop)
                                                    agent -> update_profile -> END
                       }
    agent -> max_iterations -> END   (iteration cap)

Episodic memory (Task 2a): a checkpointer (SqliteSaver in production)
persists ``messages`` per ``thread_id``. The same ``--session`` ID
restores the same conversation across restarts.

Semantic memory (Task 2b): a per-user markdown profile is loaded into
the agent's system prompt every turn, and an ``update_profile`` node
refreshes it after each final answer.
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
from .profile import build_profile_updater, load_profile, save_profile
from .router import RouteDecision, build_router
from .tools import ALL_TOOLS


class AgentState(TypedDict):
    """Mutable state passed between graph nodes."""

    messages: Annotated[list[BaseMessage], add_messages]
    classification: Optional[str]
    route_reason: Optional[str]
    iterations: int
    user_id: Optional[str]


_AGENT_SYSTEM_TEMPLATE = """You are a data analyst for the Bitext
Customer Service Tagged Training Dataset.

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
  write a concise summary of the *responses* (or *instructions*,
  depending on the question) drawing on those rows. Do not invent
  details that are not present in the rows you fetched.
- For follow-up questions referring to earlier turns ("show me 3 more",
  "what about refunds?", "total of the last two"), use the conversation
  history above to resolve what the user means before choosing a tool.
- Never answer questions that are NOT about this dataset from general
  knowledge.

WHAT YOU REMEMBER ABOUT THE CURRENT USER (their persistent profile):
---
{profile}
---

If the user asks a meta-question about themselves ("what do you
remember about me?", "what do I usually ask about?"), answer directly
from the profile above WITHOUT calling tools.

When you have enough information, produce a final answer message with
no tool calls.
"""


def _build_system_prompt(user_id: str | None) -> str:
    profile_md = load_profile(user_id or "default")
    return _AGENT_SYSTEM_TEMPLATE.format(profile=profile_md.strip())


def _strip_system_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    return [m for m in messages if not isinstance(m, SystemMessage)]


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
        system = SystemMessage(content=_build_system_prompt(state.get("user_id")))
        non_system = _strip_system_messages(state["messages"])
        response = model.invoke([system, *non_system])
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


def _make_update_profile_node():
    updater = build_profile_updater()

    def update_profile_node(state: AgentState) -> dict:
        user_id = state.get("user_id") or "default"
        messages = state["messages"]
        last_human = next(
            (m for m in reversed(messages) if isinstance(m, HumanMessage)),
            None,
        )
        last_ai = next(
            (
                m
                for m in reversed(messages)
                if isinstance(m, AIMessage) and not m.tool_calls and m.content
            ),
            None,
        )
        if last_human is None or last_ai is None:
            return {}

        current = load_profile(user_id)
        try:
            updated = updater.invoke(
                {
                    "current_profile": current,
                    "user_message": last_human.content,
                    "agent_response": last_ai.content,
                }
            )
            new_md = updated.content if hasattr(updated, "content") else str(updated)
            if new_md.strip() and new_md.strip() != current.strip():
                save_profile(user_id, new_md)
        except Exception:
            # Profile update is best-effort; never break the user-facing
            # answer because the updater hiccuped.
            pass
        return {}

    return update_profile_node


def _route_after_router(
    state: AgentState,
) -> Literal["agent", "decline"]:
    return "decline" if state.get("classification") == "out_of_scope" else "agent"


def _route_after_agent(
    state: AgentState,
) -> Literal["tools", "max_iterations", "update_profile"]:
    last = state["messages"][-1]
    has_tool_calls = (
        isinstance(last, AIMessage) and bool(getattr(last, "tool_calls", []))
    )
    if not has_tool_calls:
        return "update_profile"
    if state.get("iterations", 0) >= SETTINGS.max_iterations:
        return "max_iterations"
    return "tools"


def build_graph(checkpointer=None):
    """Compile and return the agent graph.

    Args:
        checkpointer: Optional LangGraph checkpointer for episodic memory.
            Pass a ``SqliteSaver`` instance to persist conversations across
            restarts; pass ``None`` for a stateless run.
    """
    builder: StateGraph = StateGraph(AgentState)
    builder.add_node("router", _make_router_node())
    builder.add_node("decline", _decline_node)
    builder.add_node("agent", _make_agent_node())
    builder.add_node("tools", ToolNode(ALL_TOOLS))
    builder.add_node("max_iterations", _max_iterations_node)
    builder.add_node("update_profile", _make_update_profile_node())

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
        {
            "tools": "tools",
            "max_iterations": "max_iterations",
            "update_profile": "update_profile",
        },
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("max_iterations", END)
    builder.add_edge("update_profile", END)

    return builder.compile(checkpointer=checkpointer)


__all__ = ["AgentState", "build_graph"]

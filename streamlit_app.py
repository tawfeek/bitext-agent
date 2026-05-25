"""Streamlit chat UI for the Customer Service Data Analyst Agent.

Run from the repo root:

    streamlit run streamlit_app.py

The UI shares its SQLite checkpoint file and per-user profile directory
with the CLI (see ``main.py``), so a conversation started in one can be
continued in the other under the same session ID.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from src.agent import build_graph
from src.config import SETTINGS
from src.profile import load_profile

st.set_page_config(
    page_title="Bitext Data Analyst Agent",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cached graph + checkpointer (shared across reruns)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Initializing agent…")
def get_resources() -> tuple[Any, SqliteSaver, sqlite3.Connection]:
    """Build the graph once per Streamlit process and cache it."""
    conn = sqlite3.connect(
        str(SETTINGS.checkpoint_db_path), check_same_thread=False
    )
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()
    graph = build_graph(checkpointer=checkpointer)
    return graph, checkpointer, conn


def list_known_sessions(checkpointer: SqliteSaver) -> list[str]:
    """Return every thread_id that has at least one checkpoint stored."""
    seen: set[str] = set()
    for tup in checkpointer.list(config=None):
        tid = tup.config.get("configurable", {}).get("thread_id")
        if tid:
            seen.add(tid)
    return sorted(seen)


# ---------------------------------------------------------------------------
# Helpers for rendering reasoning steps
# ---------------------------------------------------------------------------


def _truncate(text: str, limit: int = 1500) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [truncated, {len(text) - limit} more chars]"


def _format(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except TypeError:
        return str(value)


def _split_assistant_block(
    block: list[BaseMessage],
) -> tuple[list[BaseMessage], str | None]:
    """Separate a run of post-user messages into reasoning steps + final text."""
    reasoning: list[BaseMessage] = []
    final_text: str | None = None
    for m in block:
        if (
            isinstance(m, AIMessage)
            and not m.tool_calls
            and m.content
        ):
            final_text = str(m.content)
        else:
            reasoning.append(m)
    return reasoning, final_text


def render_reasoning(reasoning: list[BaseMessage]) -> None:
    """Render tool calls + observations inside an expander."""
    if not reasoning:
        return
    n_calls = sum(
        len(m.tool_calls)
        for m in reasoning
        if isinstance(m, AIMessage) and m.tool_calls
    )
    label = f"🧠 reasoning — {n_calls} tool call{'s' if n_calls != 1 else ''}"
    with st.expander(label, expanded=False):
        for m in reasoning:
            if isinstance(m, AIMessage) and m.tool_calls:
                for call in m.tool_calls:
                    st.markdown(f"**🔧 `{call['name']}`**")
                    st.code(_format(call.get("args", {})), language="json")
            elif isinstance(m, ToolMessage):
                st.markdown(f"**📤 result of `{m.name}`**")
                st.code(_truncate(_format(m.content)), language="json")


def render_history(messages: list[BaseMessage]) -> None:
    """Render the entire stored conversation as chat messages."""
    block: list[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            if block:
                with st.chat_message("assistant"):
                    reasoning, final_text = _split_assistant_block(block)
                    render_reasoning(reasoning)
                    if final_text:
                        st.markdown(final_text)
                block = []
            with st.chat_message("user"):
                st.markdown(msg.content)
        else:
            block.append(msg)
    if block:
        with st.chat_message("assistant"):
            reasoning, final_text = _split_assistant_block(block)
            render_reasoning(reasoning)
            if final_text:
                st.markdown(final_text)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def sidebar(checkpointer: SqliteSaver) -> tuple[str, str]:
    st.sidebar.title("⚙️ Session")

    known = list_known_sessions(checkpointer)

    # Sane defaults stored in session state
    if "session_id" not in st.session_state:
        st.session_state.session_id = "default"
    if "user_id" not in st.session_state:
        st.session_state.user_id = "default"

    # Quick switcher for existing sessions
    if known:
        options = ["(type a new one below)"] + known
        try:
            idx = options.index(st.session_state.session_id)
        except ValueError:
            idx = 0
        choice = st.sidebar.selectbox(
            "Existing sessions",
            options,
            index=idx,
            help="Pick a prior session to resume it. Its full message "
            "history loads from the SQLite checkpoint.",
        )
        if choice != "(type a new one below)":
            st.session_state.session_id = choice

    session_id = st.sidebar.text_input(
        "Session ID",
        value=st.session_state.session_id,
        help="Same ID = same conversation thread, persisted across "
        "restarts.",
    )
    user_id = st.sidebar.text_input(
        "User ID",
        value=st.session_state.user_id,
        help="Selects which markdown profile to load/update. Defaults "
        "to the session ID.",
    )
    st.session_state.session_id = session_id or "default"
    st.session_state.user_id = user_id or session_id or "default"

    st.sidebar.divider()

    if st.sidebar.button("🆕 New session", use_container_width=True):
        import time

        st.session_state.session_id = f"session-{int(time.time())}"
        st.rerun()

    with st.sidebar.expander("👤 User profile", expanded=False):
        st.code(load_profile(st.session_state.user_id), language="markdown")

    with st.sidebar.expander("ℹ️ About", expanded=False):
        st.markdown(
            f"- model: `{SETTINGS.nebius_model}`\n"
            f"- checkpoints: `{SETTINGS.checkpoint_db_path.name}`\n"
            f"- max iters: `{SETTINGS.max_iterations}`"
        )

    return st.session_state.session_id, st.session_state.user_id


# ---------------------------------------------------------------------------
# Streaming a single turn
# ---------------------------------------------------------------------------


def stream_turn(
    graph,
    user_text: str,
    session_id: str,
    user_id: str,
) -> None:
    """Run one turn of the graph, rendering steps live."""
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        status = st.status("🧠 thinking…", expanded=True)
        final_text: str | None = None
        try:
            for update in graph.stream(
                {
                    "messages": [HumanMessage(content=user_text)],
                    "iterations": 0,
                    "user_id": user_id,
                },
                config={"configurable": {"thread_id": session_id}},
                stream_mode="updates",
            ):
                for node, payload in update.items():
                    if not isinstance(payload, dict):
                        continue
                    if node == "router":
                        kind = payload.get("classification")
                        reason = payload.get("route_reason")
                        status.write(f"🧭 router → **{kind}** — {reason}")
                        continue
                    if node == "update_profile":
                        status.write("✏️ profile checked")
                        continue
                    for m in payload.get("messages", []):
                        if isinstance(m, AIMessage):
                            if m.tool_calls:
                                for call in m.tool_calls:
                                    status.write(
                                        f"🔧 `{call['name']}` "
                                        f"`{_format(call.get('args', {}))}`"
                                    )
                            elif m.content:
                                final_text = str(m.content)
                        elif isinstance(m, ToolMessage):
                            status.write(
                                f"📤 `{m.name}` → "
                                f"`{_truncate(_format(m.content), 280)}`"
                            )
        except Exception as exc:  # surface to user
            status.update(label="❌ error", state="error", expanded=True)
            status.write(str(exc))
            return

        status.update(label="✅ done", state="complete", expanded=False)
        if final_text:
            st.markdown(final_text)
        else:
            st.warning("Agent produced no final answer.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    graph, checkpointer, _conn = get_resources()
    session_id, user_id = sidebar(checkpointer)

    st.title("📊 Bitext Customer-Service Data Analyst")
    st.caption(
        f"Session **{session_id}** · User **{user_id}** · "
        f"Model `{SETTINGS.nebius_model}`"
    )

    # Render the existing conversation for this session.
    config = {"configurable": {"thread_id": session_id}}
    try:
        snapshot = graph.get_state(config)
        prior_messages = snapshot.values.get("messages", [])
    except Exception:
        prior_messages = []
    render_history(prior_messages)

    user_text = st.chat_input("Ask about the Bitext dataset…")
    if user_text:
        stream_turn(graph, user_text.strip(), session_id, user_id)


if __name__ == "__main__":
    main()

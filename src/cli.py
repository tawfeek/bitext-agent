"""Interactive CLI loop that prints the agent's reasoning steps.

Wraps the LangGraph agent in a SqliteSaver-backed checkpointer so that
conversations with the same ``--session`` ID resume across restarts
(Task 2a). The per-user profile (Task 2b) is loaded/updated by the
graph itself via ``--user``.
"""

from __future__ import annotations

import json
import textwrap
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from .agent import build_graph
from .config import SETTINGS
from .profile import load_profile

DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
RESET = "\033[0m"


def _truncate(text: str, limit: int = 600) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated, {len(text) - limit} more chars]"


def _format_value(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except TypeError:
        return str(value)


def _print_step(update: dict[str, Any]) -> str | None:
    """Render one streamed graph update; return final answer text if seen."""
    final_text: str | None = None
    for node, payload in update.items():
        if not isinstance(payload, dict):
            continue

        if node == "router":
            kind = payload.get("classification")
            reason = payload.get("route_reason")
            print(f"{DIM}[router]{RESET} {CYAN}{kind}{RESET} — {reason}")
            continue

        if node == "update_profile":
            print(f"{DIM}[profile updated]{RESET}")
            continue

        for msg in payload.get("messages", []):
            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    for call in msg.tool_calls:
                        args_str = _truncate(_format_value(call.get("args", {})), 400)
                        print(
                            f"{DIM}[agent → tool]{RESET} "
                            f"{YELLOW}{call['name']}{RESET}({args_str})"
                        )
                elif msg.content:
                    final_text = str(msg.content)
            elif isinstance(msg, ToolMessage):
                preview = _truncate(_format_value(msg.content), 600)
                print(
                    f"{DIM}[tool ← {msg.name}]{RESET}\n"
                    f"{textwrap.indent(preview, '  ')}"
                )
    return final_text


def _silent_step(update: dict[str, Any]) -> str | None:
    for _, payload in update.items():
        if not isinstance(payload, dict):
            continue
        for msg in payload.get("messages", []):
            if (
                isinstance(msg, AIMessage)
                and not msg.tool_calls
                and msg.content
            ):
                return str(msg.content)
    return None


_BANNER_HELP = (
    "Type a question, ':profile' to see your saved profile, "
    "':reset' to clear THIS session's conversation, or 'exit'."
)


def run_repl(
    session_id: str = "default",
    user_id: str | None = None,
    verbose: bool = True,
) -> None:
    """Launch the interactive REPL with persistent memory.

    Args:
        session_id: Thread ID for the LangGraph checkpointer. Same value
            on a future run resumes the same conversation.
        user_id: User identifier for the per-user profile. Defaults to
            ``session_id`` if not provided.
        verbose: If True, print every reasoning step. If False, only the
            final answer.
    """
    user_id = user_id or session_id
    db_path = str(SETTINGS.checkpoint_db_path)

    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        config: dict = {"configurable": {"thread_id": session_id}}

        print(f"{BOLD}Customer Service Data Analyst Agent{RESET}")
        print(
            f"{DIM}session={session_id}  user={user_id}  "
            f"checkpoints={db_path}{RESET}"
        )
        print(f"{DIM}{_BANNER_HELP}{RESET}\n")

        # Surface any restored conversation so the user knows it's there.
        try:
            snapshot = graph.get_state(config)
            prior = [
                m
                for m in snapshot.values.get("messages", [])
                if isinstance(m, (HumanMessage, AIMessage)) and m.content
            ]
            if prior:
                print(
                    f"{DIM}(resumed: {len(prior)} prior messages in this session){RESET}\n"
                )
        except Exception:
            pass

        while True:
            try:
                user_input = input(f"{GREEN}you ›{RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", ":q"}:
                return
            if user_input.lower() == ":profile":
                print(f"{MAGENTA}--- profile for {user_id} ---{RESET}")
                print(load_profile(user_id))
                print(f"{MAGENTA}---{RESET}\n")
                continue
            if user_input.lower() == ":reset":
                # Start a new thread under a derived ID. Old one stays on disk.
                import time

                session_id = f"{session_id}.{int(time.time())}"
                config = {"configurable": {"thread_id": session_id}}
                print(f"{DIM}New session: {session_id}{RESET}\n")
                continue

            final_answer: str | None = None
            try:
                for update in graph.stream(
                    {
                        "messages": [HumanMessage(content=user_input)],
                        "iterations": 0,
                        "user_id": user_id,
                    },
                    config=config,
                    stream_mode="updates",
                ):
                    step_answer = (
                        _print_step(update) if verbose else _silent_step(update)
                    )
                    if step_answer is not None:
                        final_answer = step_answer
            except Exception as exc:  # pragma: no cover - surface errors
                print(f"{RED}[error]{RESET} {exc}")
                continue

            print(
                f"\n{BOLD}agent ›{RESET} "
                f"{final_answer or '(no answer produced)'}\n"
            )

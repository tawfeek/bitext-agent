"""Interactive CLI loop that prints the agent's reasoning steps."""

from __future__ import annotations

import json
import textwrap
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from .agent import build_graph

DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
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
    """Render one streamed graph update; return the final answer text if seen."""
    final_text: str | None = None
    for node, payload in update.items():
        if not isinstance(payload, dict):
            continue

        if node == "router":
            kind = payload.get("classification")
            reason = payload.get("route_reason")
            print(f"{DIM}[router]{RESET} {CYAN}{kind}{RESET} — {reason}")
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


def run_repl(session_id: str = "default", verbose: bool = True) -> None:
    """Launch the interactive REPL.

    Args:
        session_id: Unused in Task 1 (no persistence). Kept for forward
            compatibility with Task 2.
        verbose: If True, print every reasoning step. If False, only the
            final answer.
    """
    graph = build_graph()
    print(f"{BOLD}Customer Service Data Analyst Agent{RESET}")
    print(f"{DIM}Type a question, or 'exit' to quit.{RESET}\n")

    while True:
        try:
            user = input(f"{GREEN}you ›{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user:
            continue
        if user.lower() in {"exit", "quit", ":q"}:
            return

        final_answer: str | None = None
        try:
            for update in graph.stream(
                {"messages": [HumanMessage(content=user)], "iterations": 0},
                stream_mode="updates",
            ):
                step_answer = _print_step(update) if verbose else _silent_step(update)
                if step_answer is not None:
                    final_answer = step_answer
        except Exception as exc:  # pragma: no cover - surface errors
            print(f"{RED}[error]{RESET} {exc}")
            continue

        print(f"\n{BOLD}agent ›{RESET} {final_answer or '(no answer produced)'}\n")


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

"""CLI entry point for the Customer Service Data Analyst Agent."""

from __future__ import annotations

import argparse

from src.cli import run_repl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Customer Service Data Analyst Agent — LangGraph ReAct agent "
            "for the Bitext dataset, with persistent conversation + user "
            "profile memory."
        )
    )
    parser.add_argument(
        "--session",
        default="default",
        help=(
            "Session ID. The same value on a future run resumes the same "
            "conversation thread (via SqliteSaver checkpointer)."
        ),
    )
    parser.add_argument(
        "--user",
        default=None,
        help=(
            "User ID for the persistent per-user profile. "
            "Defaults to --session if omitted."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-step reasoning output; show only final answers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_repl(
        session_id=args.session,
        user_id=args.user,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

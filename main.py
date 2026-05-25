"""CLI entry point for the Customer Service Data Analyst Agent."""

from __future__ import annotations

import argparse

from src.cli import run_repl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Customer Service Data Analyst Agent (Task 1)."
    )
    parser.add_argument(
        "--session",
        default="default",
        help="Session ID (used by Task 2 persistence; ignored in Task 1).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-step reasoning output; show only final answers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_repl(session_id=args.session, verbose=not args.quiet)


if __name__ == "__main__":
    main()

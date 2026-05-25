"""FastMCP server exposing the dataset tools over the Model Context Protocol.

Run with stdio (default; what Claude Desktop / mcp-inspector / our demo
client use):

    python mcp_server.py

Run with HTTP (Server-Sent Events) on a TCP port:

    python mcp_server.py --transport http --port 8765

All six analyst tools are exposed. They share their implementations with
the in-process LangGraph agent in ``src/tools.py`` — both call the same
DataFrame operations so MCP clients and the agent observe identical
behavior.
"""

from __future__ import annotations

import argparse
from typing import Optional

from fastmcp import FastMCP

from src.tools import (
    count_records as _count_records,
    get_examples as _get_examples,
    intent_distribution as _intent_distribution,
    list_categories as _list_categories,
    list_intents as _list_intents,
    search_examples as _search_examples,
)

mcp = FastMCP(
    name="bitext-data-analyst",
    instructions=(
        "Tools for analyzing the Bitext customer-service tagged training "
        "dataset. Use list_categories / list_intents first to discover "
        "valid filter values, then count_records, get_examples, or "
        "intent_distribution. Use search_examples for paraphrased free-"
        "text queries on customer messages."
    ),
)


@mcp.tool()
def list_categories() -> list[str]:
    """List every distinct category in the Bitext customer-service dataset.

    Start here when you do not yet know the category names. The returned
    list is sorted alphabetically.
    """
    return _list_categories.invoke({})


@mcp.tool()
def list_intents(category: Optional[str] = None) -> list[str]:
    """List distinct intents, optionally scoped to one category.

    Args:
        category: Optional category name to scope to (case-insensitive,
            e.g. "REFUND"). If omitted, returns intents across the whole
            dataset.
    """
    return _list_intents.invoke({"category": category})


@mcp.tool()
def count_records(
    category: Optional[str] = None, intent: Optional[str] = None
) -> dict:
    """Count rows in the dataset, optionally filtered by category/intent.

    Args:
        category: Optional category filter (case-insensitive).
        intent: Optional intent filter (case-insensitive, e.g.
            "get_refund").

    Returns:
        A dict with the resulting ``count`` and the ``filters`` actually
        applied.
    """
    return _count_records.invoke({"category": category, "intent": intent})


@mcp.tool()
def get_examples(
    category: Optional[str] = None,
    intent: Optional[str] = None,
    n: int = 5,
) -> list[dict]:
    """Fetch example rows, optionally filtered by category and/or intent.

    Args:
        category: Optional category filter (case-insensitive).
        intent: Optional intent filter (case-insensitive).
        n: How many example rows to return (1-25).

    Returns:
        List of rows, each containing ``category``, ``intent``,
        ``instruction`` (the customer message), ``response`` (the agent
        reply).
    """
    return _get_examples.invoke(
        {"category": category, "intent": intent, "n": n}
    )


@mcp.tool()
def search_examples(query: str, n: int = 5) -> list[dict]:
    """Find example rows whose customer instruction contains a phrase.

    Performs a case-insensitive substring search on the ``instruction``
    column. Use this when the user paraphrases what they want (e.g.
    "people wanting their money back") instead of naming a category or
    intent.

    Args:
        query: Free-text phrase to search for inside customer messages.
        n: How many matching example rows to return (1-25).
    """
    return _search_examples.invoke({"query": query, "n": n})


@mcp.tool()
def intent_distribution(category: Optional[str] = None) -> dict:
    """Return a count of rows per intent, optionally scoped to one category.

    Args:
        category: Optional category to scope the distribution to
            (case-insensitive). If omitted, returns counts across the
            whole dataset.

    Returns:
        Dict mapping intent name → row count, sorted by descending count.
    """
    return _intent_distribution.invoke({"category": category})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--transport",
        choices=["stdio", "http", "sse"],
        default="stdio",
        help=(
            "Transport for the server. 'stdio' (default) is what "
            "Claude Desktop, mcp-inspector, and the demo client use."
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host (only used by --transport http / sse).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Bind port (only used by --transport http / sse).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

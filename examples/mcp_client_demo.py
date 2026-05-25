"""Minimal MCP client demo.

Spawns the project's MCP server (`mcp_server.py`) over stdio, lists its
tools, then calls two of them and prints the results.

Run from the repo root:

    python examples/mcp_client_demo.py

This is the script the README points at to demonstrate that the MCP
server speaks the protocol properly.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from fastmcp import Client

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER_SCRIPT = REPO_ROOT / "mcp_server.py"


async def main() -> None:
    # Passing a path to a Python file tells FastMCP's Client to spawn it
    # as a subprocess and speak MCP over its stdio.
    async with Client(str(SERVER_SCRIPT)) as client:
        tools = await client.list_tools()
        print(f"Server exposes {len(tools)} tools:")
        for t in tools:
            first_line = (t.description or "").splitlines()[0]
            print(f"  - {t.name}: {first_line}")

        print("\nCalling list_categories()...")
        result = await client.call_tool("list_categories", {})
        print("  →", result.data)

        print("\nCalling count_records(intent='get_refund')...")
        result = await client.call_tool(
            "count_records", {"intent": "get_refund"}
        )
        print("  →", json.dumps(result.data, indent=2))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)

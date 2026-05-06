"""Launcher for the LectureMind MCP server.

Usage:

    # stdio transport (default — for Cursor / Claude Desktop local)
    python scripts/run_mcp_server.py

    # HTTP+SSE transport (for remote Agents / ChatGPT Apps)
    python scripts/run_mcp_server.py --sse --host 0.0.0.0 --port 8111

Environment:

* ``MCP_SERVER_TOKEN`` — required when ``--sse`` is used (bearer token).
* ``MCP_EXPOSE_SUMMARIZE`` — ``true`` to expose ``summarize_video`` and
  instantiate a Pipeline (pulls in the full ingest stack).  Default
  ``false``: read-only tools only.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from app.config import get_settings
from app.copilot.mcp_server import build_mcp


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="run_mcp_server",
        description="LectureMind MCP server (fastmcp)",
    )
    ap.add_argument(
        "--sse",
        action="store_true",
        help="Expose HTTP+SSE instead of stdio (requires MCP_SERVER_TOKEN).",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8111)
    ap.add_argument(
        "--enable-summarize",
        action="store_true",
        help="Register summarize_video (overrides MCP_EXPOSE_SUMMARIZE env).",
    )
    ap.add_argument("--log-level", default=None)
    return ap.parse_args(argv)


def _setup_logging(level: str | None) -> None:
    logging.basicConfig(
        level=(level or get_settings().log_level).upper(),
        format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.log_level)
    settings = get_settings()

    if args.sse and not os.environ.get("MCP_SERVER_TOKEN") and not settings.mcp_server_token:
        print(
            "ERROR: --sse requires MCP_SERVER_TOKEN (bearer token) to be set in env.",
            file=sys.stderr,
        )
        return 2

    expose = (
        True
        if args.enable_summarize
        else (settings.mcp_expose_summarize or False)
    )
    mcp = build_mcp(expose_summarize=expose)

    if args.sse:
        logging.getLogger(__name__).info(
            "Starting MCP SSE server on %s:%s (summarize=%s)",
            args.host,
            args.port,
            expose,
        )
        asyncio.run(mcp.run_http_async(host=args.host, port=args.port))
    else:
        logging.getLogger(__name__).info(
            "Starting MCP stdio server (summarize=%s)", expose
        )
        asyncio.run(mcp.run_stdio_async())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

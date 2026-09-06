#!/usr/bin/env python3
"""Record run-dmcp's declared tool schemas as a test fixture.

The loader reads field limits from what the server declares (`tools/list`)
instead of mirroring them as constants. Its unit tests therefore need a
`list_tools` response to serve, and a HAND-WRITTEN one would only ever say
what we believe the engine declares -- which is the copy we just deleted,
wearing a fixture's clothes.

So this records the real thing. The result is the only copy of the engine's
declared contract in this repository, and its diff is what makes a limit
changing upstream visible here:

    python record_tool_schemas.py --server-path ~/rpg/run-dmcp

Refresh it deliberately -- when a load starts failing on a length, when the
engine ships a release, or when `tests/test_loader.py` says the fixture is
missing a tool the loader now calls -- and read the diff before committing it.

Usage:
    python record_tool_schemas.py [--server-path PATH] [-o OUT] [--tools A B C]
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path

from load_to_run_dmcp import _resolve_server_entry

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path("tests/fixtures/run_dmcp_tool_schemas.json")

# The tools load_to_run_dmcp.py calls. Not a contract -- a default: the fixture
# only has to cover what the loader uses, and tests/test_loader.py fails if the
# loader calls a tool this file does not contain.
LOADER_TOOLS = (
    "create_game",
    "create_location",
    "connect_locations",
    "create_character",
    "create_item",
    "create_note",
)


async def record(server_path: str, tools: tuple[str, ...]) -> dict:
    """Connect to run-dmcp and read the declared input schemas for `tools`."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_entry = _resolve_server_entry(server_path)
    if not server_entry.exists():
        print(f"Error: run-dmcp not found at {server_entry}", file=sys.stderr)
        sys.exit(1)

    params = StdioServerParameters(
        command="node", args=[str(server_entry)], env=dict(os.environ)
    )
    async with (
        stdio_client(params) as (read_stream, write_stream),
        ClientSession(read_stream, write_stream) as session,
    ):
        init = await session.initialize()
        served = {tool.name: tool.input_schema for tool in (await session.list_tools()).tools}

    missing = [name for name in tools if name not in served]
    if missing:
        print(f"Error: server declares no such tool(s): {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    return {
        "_recorded": {
            "server": init.server_info.name,
            "version": init.server_info.version,
            "recorded_on": date.today().isoformat(),
            "tools_served": len(served),
            "command": "python record_tool_schemas.py --server-path <run-dmcp>",
            "note": (
                "Recorded from a live server -- do not hand-edit. These are the "
                "input schemas the loader reads its field limits from."
            ),
        },
        "tools": {name: served[name] for name in tools},
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Record run-dmcp's declared tool schemas")
    parser.add_argument("--server-path", default="~/rpg/run-dmcp",
                        help="Path to the run-dmcp checkout (default: ~/rpg/run-dmcp)")
    parser.add_argument("-o", "--output", default=str(DEFAULT_OUTPUT),
                        help=f"Where to write the fixture (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--tools", nargs="+", default=list(LOADER_TOOLS),
                        help="Tool names to record (default: the ones the loader calls)")
    args = parser.parse_args()

    fixture = asyncio.run(record(args.server_path, tuple(args.tools)))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    recorded = fixture["_recorded"]
    print(f"Recorded {len(fixture['tools'])} of {recorded['tools_served']} tools "
          f"from {recorded['server']} {recorded['version']} -> {out}")


if __name__ == "__main__":
    main()

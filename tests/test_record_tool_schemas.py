"""Tests for record_tool_schemas.py.

The recorder is what keeps tests/fixtures/run_dmcp_tool_schemas.json honest.
A fixture is only evidence about run-dmcp if it came from run-dmcp, so the
thing that fetches it has to be as trustworthy as the thing it fetches: a
recorder that silently wrote a partial or mislabelled file would leave the
loader's tests asserting against a fiction, with nothing to say so.
"""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, '.')
from record_tool_schemas import LOADER_TOOLS, record

SERVED = {
    "create_game": {"type": "object", "properties": {"name": {"type": "string", "maxLength": 200}}},
    "create_location": {"type": "object", "properties": {"name": {"type": "string", "maxLength": 200}}},
    "unrelated_tool": {"type": "object", "properties": {}},
}


def _fake_server(served=None, *, version="0.5.0"):
    """Patches in a server that serves `served` over a fake stdio session."""
    schemas = SERVED if served is None else served

    class FakeStdioClient:
        def __init__(self, params):
            pass

        async def __aenter__(self):
            return (None, None)

        async def __aexit__(self, *exc):
            return False

    class FakeSession:
        def __init__(self, *a):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def initialize(self):
            return SimpleNamespace(
                server_info=SimpleNamespace(name="dmcp", version=version)
            )

        async def list_tools(self):
            return SimpleNamespace(tools=[
                SimpleNamespace(name=name, input_schema=schema)
                for name, schema in schemas.items()
            ])

    return patch("mcp.client.stdio.stdio_client", FakeStdioClient), patch("mcp.ClientSession", FakeSession)


@pytest.fixture
def server_root(tmp_path):
    """A run-dmcp checkout with a declared, existing executable."""
    root = tmp_path / "run-dmcp"
    (root / "dist" / "bin").mkdir(parents=True)
    (root / "dist" / "bin" / "run-dmcp.js").write_text("// stub")
    (root / "package.json").write_text(json.dumps({
        "name": "run-dmcp",
        "bin": {"run-dmcp": "dist/bin/run-dmcp.js"},
        "main": "dist/index.js",
    }))
    return root


class TestRecord:
    def test_records_the_requested_schemas_and_nothing_else(self, server_root):
        stdio, session = _fake_server()
        with stdio, session:
            fixture = asyncio.run(record(str(server_root), ("create_game", "create_location")))

        assert set(fixture["tools"]) == {"create_game", "create_location"}
        assert fixture["tools"]["create_game"] == SERVED["create_game"]

    def test_records_which_server_it_came_from(self, server_root):
        """Provenance is the point: a fixture with no version is a rumour."""
        stdio, session = _fake_server(version="0.9.9")
        with stdio, session:
            fixture = asyncio.run(record(str(server_root), ("create_game",)))

        recorded = fixture["_recorded"]
        assert recorded["server"] == "dmcp"
        assert recorded["version"] == "0.9.9"
        assert recorded["tools_served"] == len(SERVED)
        assert recorded["recorded_on"]

    def test_refuses_to_record_a_tool_the_server_does_not_serve(self, server_root, capsys):
        """Half a fixture written silently would be worse than none."""
        stdio, session = _fake_server()
        with stdio, session, pytest.raises(SystemExit) as exit_info:
            asyncio.run(record(str(server_root), ("create_game", "invent_a_tool")))

        assert exit_info.value.code == 1
        assert "invent_a_tool" in capsys.readouterr().err

    def test_refuses_when_the_server_is_not_built(self, tmp_path, capsys):
        root = tmp_path / "run-dmcp"
        root.mkdir()
        (root / "package.json").write_text(json.dumps({"bin": "dist/bin/run-dmcp.js"}))

        stdio, session = _fake_server()
        with stdio, session, pytest.raises(SystemExit) as exit_info:
            asyncio.run(record(str(root), ("create_game",)))

        assert exit_info.value.code == 1
        assert "not found" in capsys.readouterr().err


class TestLoaderToolsDefault:
    def test_the_default_list_is_what_the_fixture_holds(self):
        """The recorded fixture and the recorder's default must not drift apart.

        tests/test_loader.py already fails if the LOADER calls a tool the
        fixture is missing; this catches the other direction, where the
        default is edited and the next refresh silently drops a tool.
        """
        recorded = json.loads(
            (Path(__file__).parent / "fixtures" / "run_dmcp_tool_schemas.json").read_text()
        )
        assert set(LOADER_TOOLS) == set(recorded["tools"])

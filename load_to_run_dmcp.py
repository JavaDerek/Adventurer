#!/usr/bin/env python3
"""
Load Adventurer JSON into run-dmcp via the MCP protocol.

Usage:
    python load_to_run_dmcp.py path/to/game.json [--server-path /path/to/run-dmcp]
    python load_to_run_dmcp.py tests/fixtures/crime_and_punishment_gold.json

Requires:
    - run-dmcp built (https://github.com/JavaDerek/run-dmcp)
    - mcp Python package: pip install mcp

See docs/RUN_DMCP_INTEGRATION.md for the tool-by-tool mapping and why each
call is shaped the way it is.
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Mirrors of run-dmcp's src/utils/validation.ts LIMITS. Fields longer than
# these are rejected by the server's zod schemas, which would abort a load
# partway through -- clamp locally instead.
NAME_MAX = 200
DESCRIPTION_MAX = 5000
CONTENT_MAX = 50000


def truncate(text: str | None, limit: int) -> str:
    """Clamp text to a run-dmcp field limit."""
    if not text:
        return ""
    return text if len(text) <= limit else text[:limit]


def exit_direction(destination_name: str) -> str:
    """Build the direction label for an exit leading to `destination_name`.

    run-dmcp stores exits keyed by direction and drops any existing exit that
    shares a direction with a new one, so every exit out of a room needs a
    distinct label. Adventurer's source JSON carries no compass data -- exits
    are just destination names -- so the destination is what makes the label
    unique, and it reads naturally when narrated.
    """
    return truncate(f"toward {destination_name}", NAME_MAX)


async def load_game_with_session(
    session,
    game_data: dict,
    setting: str = "interactive fiction",
    style: str = "exploration and puzzle-solving"
) -> dict:
    """Load game data into run-dmcp using an existing MCP session.

    This is the core logic, separated for testability.
    """
    reset_schema_drift_state()

    title = truncate(game_data.get("title", "Untitled Adventure"), NAME_MAX)
    rooms = game_data.get("rooms", [])

    # Create the game
    print("\nCreating game...")
    game_result = await call_tool(session, "create_game", {
        "name": title,
        "setting": truncate(setting, DESCRIPTION_MAX),
        "style": truncate(style, NAME_MAX),
    })
    game_id = extract_id_from_result(game_result, "game")
    web_ui_url = extract_web_ui_url(game_result)
    print(f"  Game ID: {game_id}")

    # Create all locations first (need IDs for connections)
    print("\nCreating locations...")
    location_ids = {}
    for room in rooms:
        name = truncate(room.get("name", "Unknown Room"), NAME_MAX)
        description = truncate(room.get("description", ""), DESCRIPTION_MAX)
        atmosphere = room.get("atmosphere", "")

        args = {
            "gameId": game_id,
            "name": name,
            "description": description,
        }
        # run-dmcp models atmosphere as a first-class location property, so it
        # no longer has to be appended to the prose description.
        if atmosphere:
            args["properties"] = {"atmosphere": truncate(atmosphere, DESCRIPTION_MAX)}

        result = await call_tool(session, "create_location", args)
        loc_id = extract_id_from_result(result, "location")
        location_ids[room.get("name", "Unknown Room")] = loc_id
        print(f"  + {name}")

    # Connect locations based on exits
    print("\nConnecting locations...")
    connections_made = 0
    for room in rooms:
        source_name = room.get("name", "")
        source_id = location_ids.get(source_name)
        if not source_id:
            continue

        seen_exits = set()
        for exit_name in room.get("exits", []):
            if exit_name == source_name:
                logger.warning(
                    "Skipping self-referential exit in room '%s'", source_name
                )
                continue
            if exit_name in seen_exits:
                continue
            seen_exits.add(exit_name)

            target_id = location_ids.get(exit_name)
            if not target_id:
                print(f"  Warning: Exit '{exit_name}' not found (from {source_name})")
                continue

            try:
                # bidirectional=False on purpose: run-dmcp defaults it to True,
                # but the source JSON already lists the reverse exit whenever
                # one exists (fix_exits.py adds it). Letting the server infer
                # reverse exits would invent connections the map never had and
                # silently repair the one-way doors analyze_map.py reports on.
                await call_tool(session, "connect_locations", {
                    "fromLocationId": source_id,
                    "toLocationId": target_id,
                    "fromDirection": exit_direction(exit_name),
                    "toDirection": exit_direction(source_name),
                    "bidirectional": False,
                })
                connections_made += 1
            except Exception:
                logger.exception(
                    "connect_locations failed: %s -> %s", source_name, exit_name
                )
                print(f"  Warning: Could not connect {source_name} -> {exit_name}")
    print(f"  {connections_made} connections created")

    # Collect and deduplicate characters
    print("\nCreating characters...")
    character_locations = {}  # character_name -> first location
    for room in rooms:
        room_name = room.get("name", "")
        for char in room.get("characters", []):
            # Normalize character name (remove parentheticals)
            normalized = normalize_character_name(char)
            if normalized not in character_locations:
                character_locations[normalized] = room_name

    character_ids = {}
    for char_name, first_location in character_locations.items():
        loc_id = location_ids.get(first_location)
        try:
            result = await call_tool(session, "create_character", {
                "gameId": game_id,
                "name": truncate(char_name, NAME_MAX),
                "isPlayer": char_name.lower() == "player",
                "locationId": loc_id,
            })
        except Exception:
            logger.exception("create_character failed for '%s'", char_name)
            print(f"  Warning: Could not create character '{char_name}'")
            continue
        character_ids[char_name] = extract_id_from_result(result, "character")
        print(f"  + {char_name} (at {first_location})")

    # Create items at their locations
    print("\nCreating items...")
    items_created = 0
    for room in rooms:
        room_name = room.get("name", "")
        loc_id = location_ids.get(room_name)
        if not loc_id:
            continue

        for item_name in room.get("items", []):
            try:
                await call_tool(session, "create_item", {
                    "gameId": game_id,
                    "ownerId": loc_id,
                    "ownerType": "location",
                    "name": truncate(item_name, NAME_MAX),
                    "properties": {
                        "description": truncate(f"Found in {room_name}", DESCRIPTION_MAX),
                    },
                })
                items_created += 1
            except Exception:
                logger.exception(
                    "create_item failed for '%s' in '%s'", item_name, room_name
                )
                print(f"  Warning: Could not create item '{item_name}'")
    print(f"  {items_created} items created")

    # Record the source material's beats as GM notes.
    #
    # These are things that happen in the *source text*, not things that have
    # happened in this playthrough. run-dmcp's log_event writes narrative
    # history, and seeding unplayed beats there would tell the narrator they
    # already occurred. A note attached to the location is the honest home for
    # them: available as reference, not asserted as history.
    print("\nRecording source beats as notes...")
    notes_created = 0
    beats_recorded = 0
    for room in rooms:
        room_name = room.get("name", "")
        events = room.get("events", [])
        if not events:
            continue
        loc_id = location_ids.get(room_name)

        content = "\n".join(f"- {event}" for event in events)
        try:
            await call_tool(session, "create_note", {
                "gameId": game_id,
                "title": truncate(f"Source beats: {room_name}", NAME_MAX),
                "content": truncate(content, CONTENT_MAX),
                "category": "plot",
                "relatedEntityId": loc_id,
                "relatedEntityType": "location",
                "tags": ["adventurer", "source-beat"],
            })
            notes_created += 1
            beats_recorded += len(events)
        except Exception:
            logger.exception("create_note failed for room '%s'", room_name)
            print(f"  Warning: Could not record beats for '{room_name}'")
    print(f"  {notes_created} notes covering {beats_recorded} beats")

    # Summary
    print("\n" + "=" * 50)
    print("LOAD COMPLETE")
    print("=" * 50)
    print(f"Game ID: {game_id}")
    print(f"Title: {title}")
    print(f"Locations: {len(location_ids)}")
    print(f"Connections: {connections_made}")
    print(f"Characters: {len(character_ids)}")
    print(f"Items: {items_created}")
    print(f"Notes: {notes_created} ({beats_recorded} beats)")
    if web_ui_url:
        print(f"\nWeb UI: {web_ui_url}")
    print(
        f"\nTo play: point an MCP client at run-dmcp, then use load_game with ID: {game_id}"
    )

    return {
        "game_id": game_id,
        "title": title,
        "web_ui_url": web_ui_url,
        "location_ids": location_ids,
        "connections_made": connections_made,
        "character_ids": character_ids,
        "items_created": items_created,
        "notes_created": notes_created,
        "beats_recorded": beats_recorded,
    }


# Hook names the MCP Python SDK has used for output-schema validation. The
# public name is `validate_tool_result` (mcp >= 2.0); `_validate_tool_result`
# was the private predecessor. Patching a name that no longer exists fails
# silently -- the import below just adds an unused attribute -- so try both and
# insist that at least one landed.
_VALIDATION_HOOKS = ("validate_tool_result", "_validate_tool_result")


def _disable_structured_output_validation(session_cls) -> list:
    """Stop the MCP SDK from rejecting results that don't match a declared schema.

    Called only after a server has actually returned content its own declared
    schema rejects. Validation is otherwise left enabled, so genuine drift in a
    future run-dmcp surfaces instead of being silently suppressed.

    Returns the hook names that were patched, so callers can tell whether the
    SDK still exposes one.
    """
    async def noop_validate(self, name, result):
        return None

    patched = []
    for hook in _VALIDATION_HOOKS:
        if hasattr(session_cls, hook):
            setattr(session_cls, hook, noop_validate)
            patched.append(hook)

    if not patched:
        logger.warning(
            "No known MCP output-validation hook found (tried %s); a schema "
            "mismatch from the server may abort the load.",
            ", ".join(_VALIDATION_HOOKS),
        )
    return patched


# A server whose declared outputSchema disagrees with the payload it actually
# returns. run-dmcp shipped exactly this on its four character tools
# (https://github.com/JavaDerek/run-dmcp/issues/24, fixed upstream), which made
# every character silently fail while locations and items succeeded.
_SCHEMA_DRIFT_MARKERS = (
    "invalid structured content",
    "additional properties are not allowed",
    "did not return structured content",
)

_schema_drift_detected = False


def is_schema_drift_error(exc: BaseException) -> bool:
    """True if `exc` is the SDK rejecting a result against its declared schema."""
    message = str(exc).lower()
    return any(marker in message for marker in _SCHEMA_DRIFT_MARKERS)


def schema_drift_detected() -> bool:
    """Whether this run has degraded output validation."""
    return _schema_drift_detected


def reset_schema_drift_state() -> None:
    """Clear the degrade latch. Called at the start of each load, and by tests."""
    global _schema_drift_detected
    _schema_drift_detected = False


async def call_tool(session, name: str, arguments: dict):
    """Call an MCP tool, tolerating a server that violates its own output schema.

    Output validation is left enabled, so a genuine mismatch stays visible. The
    first time a server actually returns content its declared schema rejects,
    this explains the problem once, turns validation off for the rest of the
    run, and retries -- the payload is well-formed, and a load should not lose
    every character over a wrong declaration.
    """
    global _schema_drift_detected
    try:
        return await session.call_tool(name, arguments)
    except RuntimeError as exc:
        if not is_schema_drift_error(exc):
            raise
        if not _schema_drift_detected:
            _schema_drift_detected = True
            logger.warning(
                "Tool '%s' returned content that its own declared output schema "
                "rejects (%s). This is a server-side bug: run-dmcp had it on the "
                "character tools until "
                "https://github.com/JavaDerek/run-dmcp/issues/24 was fixed, so an "
                "outdated run-dmcp build is the likely cause -- pull and rebuild "
                "it. Disabling output validation for the rest of this load and "
                "continuing; the payloads themselves are intact.",
                name,
                exc,
            )
            _disable_structured_output_validation(type(session))
        return await session.call_tool(name, arguments)


def _read_game_json(json_path: str) -> dict:
    """Read an Adventurer JSON file from disk."""
    with open(json_path, "r") as f:
        return json.load(f)


# run-dmcp's executable entry, if its package.json cannot be read. Only a
# fallback: the declaration is the authority, and see _resolve_server_entry.
DEFAULT_SERVER_ENTRY = "dist/bin/run-dmcp.js"


def _declared_bin(bin_field: Any) -> str | None:
    """The path npm would install as an executable, from a `bin` field.

    npm allows a bare string for a single executable, or a name -> path map.
    A map with several entries is only unambiguous if one is named for the
    package itself; anything else is a manifest we should not guess about.
    """
    if isinstance(bin_field, str):
        return bin_field
    if isinstance(bin_field, dict):
        if isinstance(bin_field.get("run-dmcp"), str):
            return bin_field["run-dmcp"]
        if len(bin_field) == 1:
            only = next(iter(bin_field.values()))
            if isinstance(only, str):
                return only
    return None


def _resolve_server_entry(server_path: str) -> Path:
    """Resolve run-dmcp's executable from the engine's own declaration.

    NOT a remembered path. run-dmcp split library from application on
    2026-08-29 (commit 3232645, "importing the package started an HTTP server
    and squatted a port"): before it, `dist/index.js` bound a port and
    connected a stdio transport as a side effect of import, so spawning it
    worked; after it, that file is exports and nothing else. The engine's own
    commit message says a config pointing at it now starts nothing -- and this
    loader was such a config for six days, because node runs the file, it
    exits, and `session.initialize()` dies with "Connection closed" before a
    single tool call. No mock of a tool can see a handshake that never happens.

    So read `package.json`, which names both halves and ships in every
    checkout: `bin` is the application, `main` is the library. Hardcoding
    today's answer would only catch an edit back toward the library entry,
    which nobody will make, and would sail straight through the engine moving
    its executable again -- which is the failure that actually happened.
    """
    root = Path(server_path).expanduser()
    manifest_path = root / "package.json"

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning(
            "Could not read %s (%s); falling back to %s. If run-dmcp has moved "
            "its executable, this will fail at session.initialize().",
            manifest_path, exc, DEFAULT_SERVER_ENTRY,
        )
        print(f"Warning: no readable package.json at {manifest_path}; "
              f"assuming {DEFAULT_SERVER_ENTRY}")
        return root / DEFAULT_SERVER_ENTRY

    declared = _declared_bin(manifest.get("bin"))
    if declared is None:
        logger.warning(
            "%s declares no unambiguous 'bin'; falling back to %s.",
            manifest_path, DEFAULT_SERVER_ENTRY,
        )
        print(f"Warning: {manifest_path} declares no executable; "
              f"assuming {DEFAULT_SERVER_ENTRY}")
        return root / DEFAULT_SERVER_ENTRY

    entry = root / declared
    library = manifest.get("main")
    if isinstance(library, str) and (root / library).resolve() == entry.resolve():
        logger.error(
            "%s declares the same file as both 'bin' and 'main' (%s). The "
            "library entry starts no server; spawning it would fail at "
            "session.initialize() with 'Connection closed'.",
            manifest_path, declared,
        )
        print(f"Error: {manifest_path} names {declared} as both its executable "
              "and its library entry; the library entry starts nothing.")
        sys.exit(1)

    return entry


async def load_game_to_run_dmcp(
    json_path: str,
    server_path: str,
    setting: str = "interactive fiction",
    style: str = "exploration and puzzle-solving"
) -> dict:
    """Load an Adventurer JSON file into run-dmcp.

    This is the entry point that handles MCP connection setup.
    """
    # Import MCP SDK
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError:
        print("Error: mcp package not installed. Run: pip install mcp")
        sys.exit(1)


    # Load JSON data (synchronously, before the server connection is opened)
    game_data = await asyncio.to_thread(_read_game_json, json_path)

    print(f"Loading: {game_data.get('title', 'Untitled')}")
    print(f"Rooms: {len(game_data.get('rooms', []))}")

    server_entry = _resolve_server_entry(server_path)
    if not server_entry.exists():
        print(f"Error: run-dmcp not found at {server_entry}")
        print("Install it: git clone https://github.com/JavaDerek/run-dmcp.git")
        print("Then: cd run-dmcp && npm ci && cd client && npm ci && cd .. && npm run build")
        sys.exit(1)

    # Connect to run-dmcp via MCP protocol.
    #
    # env must be passed explicitly: the MCP SDK launches the child with a
    # scrubbed default environment, so without this the server never sees
    # DMCP_DB_PATH / DMCP_HTTP_PORT / DMCP_LOG_LEVEL and quietly writes to its
    # default database instead of the one the operator asked for.
    server_params = StdioServerParameters(
        command="node",
        args=[str(server_entry)],
        env=dict(os.environ),
    )

    async with (
        stdio_client(server_params) as (read_stream, write_stream),
        ClientSession(read_stream, write_stream) as session,
    ):
        await session.initialize()
        return await load_game_with_session(session, game_data, setting, style)


def _result_json(result: Any) -> dict | None:
    """Parse the first JSON object out of an MCP tool result, if there is one."""
    if not hasattr(result, "content"):
        return None
    for content in result.content:
        if hasattr(content, "text"):
            try:
                data = json.loads(content.text)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(data, dict):
                return data
    return None


def extract_id_from_result(result: Any, entity_type: str) -> str:
    """Extract an ID from an MCP tool result."""
    # MCP results come as content list
    if hasattr(result, 'content'):
        for content in result.content:
            if hasattr(content, 'text'):
                text = content.text
                # Try to parse as JSON
                try:
                    data = json.loads(text)
                    # Look for common ID field names
                    for key in ['id', f'{entity_type}Id', f'{entity_type}_id', 'gameId', 'locationId', 'characterId']:
                        if key in data:
                            return data[key]
                    # If it's a nested object
                    if entity_type in data and 'id' in data[entity_type]:
                        return data[entity_type]['id']
                except (json.JSONDecodeError, TypeError):
                    # Try regex extraction
                    match = re.search(r'"id"\s*:\s*"([^"]+)"', text)
                    if match:
                        return match.group(1)

    # Fallback: return the raw result as string
    return str(result)


def extract_web_ui_url(result: Any) -> str | None:
    """Pull run-dmcp's web UI URL out of a create_game result, if present."""
    data = _result_json(result)
    if not data:
        return None
    web_ui = data.get("webUi")
    if isinstance(web_ui, dict):
        url = web_ui.get("url")
        if isinstance(url, str):
            return url
    return None


def normalize_character_name(name: str) -> str:
    """Normalize character name by removing parentheticals and extra whitespace."""
    # Remove parenthetical suffixes like "(briefly)", "(via viewscreen)", etc.
    normalized = re.sub(r'\s*\([^)]*\)\s*', '', name)
    # Clean up whitespace
    normalized = ' '.join(normalized.split())
    return normalized


def main():
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Load Adventurer JSON into run-dmcp for gameplay with an LLM"
    )
    parser.add_argument(
        "json_file",
        help="Path to the Adventurer JSON file"
    )
    parser.add_argument(
        "--server-path",
        default=os.environ.get("RUN_DMCP_PATH", "~/rpg/run-dmcp"),
        help="Path to the run-dmcp checkout (default: $RUN_DMCP_PATH or ~/rpg/run-dmcp)"
    )
    parser.add_argument(
        "--setting",
        default="interactive fiction",
        help="Game setting description (default: 'interactive fiction')"
    )
    parser.add_argument(
        "--style",
        default="exploration and puzzle-solving",
        help="Game style description (default: 'exploration and puzzle-solving')"
    )

    args = parser.parse_args()

    if not os.path.exists(args.json_file):
        print(f"Error: File not found: {args.json_file}")
        sys.exit(1)

    try:
        asyncio.run(load_game_to_run_dmcp(
            json_path=args.json_file,
            server_path=args.server_path,
            setting=args.setting,
            style=args.style
        ))
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(1)
    except SystemExit:
        raise
    except Exception as e:
        logger.exception("Load failed")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Load Adventurer JSON into DMCP via MCP protocol.

Usage:
    python load_to_dmcp.py path/to/game.json [--dmcp-path /path/to/dmcp]
    python load_to_dmcp.py tests/fixtures/deadly_assassin_gold.json

Requires:
    - DMCP installed (https://github.com/shawnrushefsky/dmcp)
    - mcp Python package: pip install mcp
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


async def load_game_to_dmcp(
    json_path: str,
    dmcp_path: str,
    setting: str = "interactive fiction",
    style: str = "exploration and puzzle-solving"
) -> dict:
    """Load an Adventurer JSON file into DMCP."""

    # Import MCP SDK
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError:
        print("Error: mcp package not installed. Run: pip install mcp")
        sys.exit(1)

    # Disable strict output validation (DMCP returns extra fields the SDK doesn't expect)
    import mcp.client.session as mcp_session
    async def noop_validate(self, name, result):
        pass
    mcp_session.ClientSession._validate_tool_result = noop_validate

    # Load JSON data
    with open(json_path, "r") as f:
        game_data = json.load(f)

    title = game_data.get("title", "Untitled Adventure")
    rooms = game_data.get("rooms", [])

    print(f"Loading: {title}")
    print(f"Rooms: {len(rooms)}")

    # Resolve DMCP path
    dmcp_index = Path(dmcp_path).expanduser() / "dist" / "index.js"
    if not dmcp_index.exists():
        print(f"Error: DMCP not found at {dmcp_index}")
        print("Please install DMCP: git clone https://github.com/shawnrushefsky/dmcp.git")
        print("Then: cd dmcp && npm install && npm run build")
        sys.exit(1)

    # Connect to DMCP via MCP protocol
    server_params = StdioServerParameters(
        command="node",
        args=[str(dmcp_index)],
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Create the game
            print("\nCreating game...")
            result = await session.call_tool("create_game", {
                "name": title,
                "setting": setting,
                "style": style
            })
            game_id = extract_id_from_result(result, "game")
            print(f"  Game ID: {game_id}")

            # Create all locations first (need IDs for connections)
            print("\nCreating locations...")
            location_ids = {}
            for room in rooms:
                name = room.get("name", "Unknown Room")
                description = room.get("description", "")
                atmosphere = room.get("atmosphere", "")

                # Append atmosphere to description
                if atmosphere:
                    description = f"{description}\n\nAtmosphere: {atmosphere}"

                result = await session.call_tool("create_location", {
                    "gameId": game_id,
                    "name": name,
                    "description": description
                })
                loc_id = extract_id_from_result(result, "location")
                location_ids[name] = loc_id
                print(f"  + {name}")

            # Connect locations based on exits
            print("\nConnecting locations...")
            connections_made = 0
            for room in rooms:
                source_name = room.get("name", "")
                source_id = location_ids.get(source_name)
                if not source_id:
                    continue

                for exit_name in room.get("exits", []):
                    target_id = location_ids.get(exit_name)
                    if target_id:
                        await session.call_tool("connect_locations", {
                            "gameId": game_id,
                            "fromId": source_id,
                            "toId": target_id
                        })
                        connections_made += 1
                    else:
                        print(f"  Warning: Exit '{exit_name}' not found (from {source_name})")
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
                result = await session.call_tool("create_character", {
                    "gameId": game_id,
                    "name": char_name,
                    "isPlayer": char_name.lower() == "player",
                    "locationId": loc_id
                })
                char_id = extract_id_from_result(result, "character")
                character_ids[char_name] = char_id
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
                        await session.call_tool("create_item", {
                            "gameId": game_id,
                            "name": item_name,
                            "description": f"Found in {room_name}",
                            "locationId": loc_id
                        })
                        items_created += 1
                    except Exception as e:
                        print(f"  Warning: Could not create item '{item_name}': {e}")
            print(f"  {items_created} items created")

            # Add events as narrative context
            print("\nAdding events...")
            events_added = 0
            for room in rooms:
                room_name = room.get("name", "")
                loc_id = location_ids.get(room_name)

                for event in room.get("events", []):
                    try:
                        await session.call_tool("add_event", {
                            "gameId": game_id,
                            "description": f"[{room_name}] {event}",
                            "locationId": loc_id
                        })
                        events_added += 1
                    except Exception as e:
                        # add_event might have different signature, try without locationId
                        try:
                            await session.call_tool("add_event", {
                                "gameId": game_id,
                                "description": f"[{room_name}] {event}"
                            })
                            events_added += 1
                        except Exception:
                            pass  # Skip if events aren't supported this way
            print(f"  {events_added} events added")

            # Summary
            print("\n" + "=" * 50)
            print("LOAD COMPLETE")
            print("=" * 50)
            print(f"Game ID: {game_id}")
            print(f"Title: {title}")
            print(f"Locations: {len(location_ids)}")
            print(f"Characters: {len(character_ids)}")
            print(f"Items: {items_created}")
            print(f"Events: {events_added}")
            print(f"\nTo play: Configure Claude Desktop with DMCP, then use load_game with ID: {game_id}")

            return {
                "game_id": game_id,
                "title": title,
                "location_ids": location_ids,
                "character_ids": character_ids,
                "items_created": items_created,
                "events_added": events_added
            }


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
                except json.JSONDecodeError:
                    # Try regex extraction
                    match = re.search(r'"id"\s*:\s*"([^"]+)"', text)
                    if match:
                        return match.group(1)

    # Fallback: return the raw result as string
    return str(result)


def normalize_character_name(name: str) -> str:
    """Normalize character name by removing parentheticals and extra whitespace."""
    # Remove parenthetical suffixes like "(briefly)", "(via viewscreen)", etc.
    normalized = re.sub(r'\s*\([^)]*\)\s*', '', name)
    # Clean up whitespace
    normalized = ' '.join(normalized.split())
    return normalized


def main():
    parser = argparse.ArgumentParser(
        description="Load Adventurer JSON into DMCP for gameplay with Claude"
    )
    parser.add_argument(
        "json_file",
        help="Path to the Adventurer JSON file"
    )
    parser.add_argument(
        "--dmcp-path",
        default="~/dmcp",
        help="Path to DMCP installation (default: ~/dmcp)"
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
        result = asyncio.run(load_game_to_dmcp(
            json_path=args.json_file,
            dmcp_path=args.dmcp_path,
            setting=args.setting,
            style=args.style
        ))
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

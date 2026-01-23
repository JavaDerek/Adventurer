#!/usr/bin/env python3
"""
Map Healer - Uses LLM to intelligently fix connectivity issues in room maps.

Takes a rooms JSON file, analyzes connectivity issues, and uses an LLM to:
- Add missing bidirectional connections
- Create transition rooms to connect disconnected areas
- Resolve broken references
- Suggest puzzle-gated paths where appropriate
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Any
from openai import OpenAI
from dotenv import load_dotenv
import httpx

from analyze_map import MapAnalyzer

# Load environment variables
load_dotenv()


HEAL_PROMPT = """You are an expert interactive fiction game designer. Your task is to heal a room map that has connectivity issues.

## Current Rooms
{rooms_json}

## Connectivity Issues Found

### Broken References (exits pointing to non-existent rooms)
{broken_refs}

### One-Way Doors (can go A→B but not B→A)
{one_way_doors}

### Unreachable Rooms (cannot be reached from {start_room})
{unreachable}

### Disconnected Areas
{subgraphs}

## Your Task

Fix this map to create a fully connected, playable interactive fiction game. You should:

1. **Add return paths** for one-way doors (most rooms should have bidirectional connections)

2. **Create transition rooms** where needed to connect disconnected areas. For example:
   - If there's a "Matrix" area disconnected from the main world, create an "APC Chamber" or similar room that connects them
   - These transition rooms should feel natural to the narrative

3. **Resolve broken references** by either:
   - Creating the missing room if it makes sense narratively
   - Removing the exit if the reference seems like an error
   - Replacing with a valid existing room name

4. **Add puzzle hints** where appropriate:
   - Some connections can require items or solving puzzles
   - Add an optional "requires" field to exits that should be locked
   - Example: {{"exit": "Matrix - Quarry", "requires": "neural_link_active"}}

5. **Ensure all rooms are reachable** from the starting room ({start_room})

## Output Format

Return a complete JSON object with the healed rooms. Use this exact structure:

```json
{{
  "rooms": [
    {{
      "name": "Room Name",
      "description": "3-5 sentence description of the room",
      "exits": ["Room A", "Room B"],
      "items": ["item1", "item2"],
      "characters": ["Character1"],
      "events": ["event description"],
      "atmosphere": "mood_word",
      "notes": "optional designer notes about changes made"
    }}
  ],
  "healing_summary": {{
    "rooms_added": ["list of new room names"],
    "connections_added": ["Room A <-> Room B", ...],
    "connections_removed": ["broken ref removed", ...],
    "puzzle_gates": ["description of puzzle requirements"]
  }}
}}
```

Important:
- Keep all existing room data intact unless fixing an issue
- Maintain the narrative tone and atmosphere of the original
- Make the game interesting but fair - every room should be reachable
- The Matrix should require some mechanism to enter (like using the APC)

Return ONLY the JSON object, no additional text."""


class MapHealer:
    """Uses LLM to intelligently heal map connectivity issues."""

    def __init__(self, api_base: str, api_key: str, model: str):
        self.client = OpenAI(
            base_url=api_base,
            api_key=api_key,
            timeout=httpx.Timeout(600.0, connect=60.0)
        )
        self.model = model

    def heal_map(self, rooms_data: dict, start_room: str = "TARDIS") -> dict:
        """Heal the map using LLM analysis."""
        # Run analyzer to get issues
        analyzer = MapAnalyzer(rooms_data)

        broken_refs = analyzer.find_broken_references()
        one_way = analyzer.find_one_way_doors()
        unreachable = analyzer.find_unreachable_rooms(start_room)
        subgraphs = analyzer.find_disconnected_subgraphs()

        # Format issues for the prompt
        broken_refs_text = "\n".join(
            f"  - {from_room} -> \"{to_room}\" (doesn't exist)"
            for from_room, to_room in broken_refs
        ) or "  None"

        one_way_text = "\n".join(
            f"  - {from_room} -> {to_room} (no return path)"
            for from_room, to_room in one_way
        ) or "  None"

        unreachable_text = "\n".join(
            f"  - {room}" for room in sorted(unreachable)
        ) or "  None"

        subgraphs_text = ""
        if len(subgraphs) > 1:
            for i, sg in enumerate(subgraphs, 1):
                label = "Matrix" if any("Matrix" in r for r in sg) else f"Area {i}"
                rooms_list = ", ".join(sorted(sg)[:5])
                if len(sg) > 5:
                    rooms_list += f", ... ({len(sg)} rooms total)"
                subgraphs_text += f"  Subgraph {i} ({label}): {rooms_list}\n"
        else:
            subgraphs_text = "  All rooms are connected"

        # Build the prompt
        prompt = HEAL_PROMPT.format(
            rooms_json=json.dumps(rooms_data, indent=2),
            broken_refs=broken_refs_text,
            one_way_doors=one_way_text,
            unreachable=unreachable_text,
            subgraphs=subgraphs_text,
            start_room=start_room
        )

        print(f"🔧 Healing map with {len(rooms_data.get('rooms', []))} rooms...")
        print(f"   Issues found: {len(broken_refs)} broken refs, {len(one_way)} one-way doors, {len(unreachable)} unreachable")

        # Call LLM
        response = self._call_llm(prompt)

        # Parse response
        healed_data = self._extract_json(response)

        return healed_data

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM and return the response."""
        print(f"🤖 Calling {self.model}...")

        # Determine token parameter name based on model
        token_param = "max_tokens"
        if any(x in self.model.lower() for x in ["gpt-5", "o1", "o3"]):
            token_param = "max_completion_tokens"

        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,  # Lower temperature for more consistent output
            token_param: 8000,
            "stream": True
        }

        # Collect streaming response
        full_response = ""
        try:
            stream = self.client.chat.completions.create(**kwargs)
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    print(".", end="", flush=True)
            print()  # Newline after dots
        except Exception as e:
            print(f"\n❌ LLM call failed: {e}")
            raise

        return full_response

    def _extract_json(self, response_text: str) -> dict:
        """Extract JSON from the LLM response."""
        # Try direct parse
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Try markdown code block
        if "```json" in response_text:
            try:
                json_text = response_text.split("```json")[1].split("```")[0].strip()
                return json.loads(json_text)
            except (json.JSONDecodeError, IndexError):
                pass

        if "```" in response_text:
            try:
                json_text = response_text.split("```")[1].split("```")[0].strip()
                return json.loads(json_text)
            except (json.JSONDecodeError, IndexError):
                pass

        # Try to find JSON object
        start = response_text.find('{')
        if start >= 0:
            depth = 0
            for i in range(start, len(response_text)):
                if response_text[i] == '{':
                    depth += 1
                elif response_text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(response_text[start:i+1])
                        except json.JSONDecodeError:
                            pass
                        break

        raise ValueError("Could not extract valid JSON from LLM response")


def get_llm_config(use_remote: bool) -> tuple:
    """Get LLM configuration based on remote/local flag."""
    if use_remote:
        api_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL", "gpt-4o")

        if not api_key or api_key == "sk-your-key-here":
            print("❌ Error: OPENAI_API_KEY not set in .env file")
            print("   Get your API key from: https://platform.openai.com/api-keys")
            sys.exit(1)

        return api_base, api_key, model, "remote (OpenAI)"
    else:
        api_base = os.getenv("LLM_BASE_URL") or os.getenv("LLM_API_BASE", "http://localhost:8000/v1")
        api_key = os.getenv("LLM_API_KEY", "lm-studio")
        model = os.getenv("LLM_MODEL", "local-model")
        return api_base, api_key, model, "local"


def main():
    parser = argparse.ArgumentParser(
        description="Heal connectivity issues in interactive fiction room maps using LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python heal_map.py rooms.json                    # Use remote OpenAI (default)
  python heal_map.py --local rooms.json            # Use local LLM
  python heal_map.py rooms.json -o healed.json     # Specify output file
  python heal_map.py rooms.json --start "TARDIS"   # Specify starting room

Environment variables:
  OpenAI:     OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL
  Local LLM:  LLM_BASE_URL, LLM_MODEL, LLM_API_KEY
"""
    )
    parser.add_argument(
        "json_file",
        help="Path to the rooms JSON file to heal"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: <input>_healed.json)"
    )
    parser.add_argument(
        "-s", "--start",
        default="TARDIS",
        help="Starting room for connectivity analysis (default: TARDIS)"
    )
    parser.add_argument(
        "--local", "-l",
        action="store_true",
        help="Use local LLM instead of OpenAI (default: use OpenAI)"
    )
    parser.add_argument(
        "--analyze-only", "-a",
        action="store_true",
        help="Only show analysis, don't heal"
    )

    args = parser.parse_args()

    # Load input file
    try:
        with open(args.json_file, "r") as f:
            rooms_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: File not found: {args.json_file}")
        return 1
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON: {e}")
        return 1

    # Show analysis
    print("=" * 50)
    print("MAP ANALYSIS (before healing)")
    print("=" * 50)
    analyzer = MapAnalyzer(rooms_data)
    print(analyzer.generate_report(args.start))

    if args.analyze_only:
        return 0

    # Get LLM config (remote by default for healing)
    use_remote = not args.local
    api_base, api_key, model, mode_name = get_llm_config(use_remote)

    print(f"\n🔗 Using {mode_name} LLM: {model}")

    # Heal the map
    healer = MapHealer(api_base, api_key, model)

    try:
        healed_data = healer.heal_map(rooms_data, args.start)
    except Exception as e:
        print(f"❌ Healing failed: {e}")
        return 1

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base = args.json_file.rsplit('.', 1)[0]
        output_path = f"{base}_healed.json"

    # Add metadata
    healed_data["original_file"] = args.json_file
    healed_data["healed_by"] = model

    # Save healed map
    with open(output_path, "w") as f:
        json.dump(healed_data, f, indent=2)

    print(f"\n✅ Healed map saved to: {output_path}")

    # Show healing summary if available
    if "healing_summary" in healed_data:
        summary = healed_data["healing_summary"]
        print("\n📋 Healing Summary:")
        if summary.get("rooms_added"):
            print(f"   Rooms added: {', '.join(summary['rooms_added'])}")
        if summary.get("connections_added"):
            print(f"   Connections added: {len(summary['connections_added'])}")
        if summary.get("puzzle_gates"):
            print(f"   Puzzle gates: {len(summary['puzzle_gates'])}")

    # Verify the healed map
    print("\n" + "=" * 50)
    print("MAP ANALYSIS (after healing)")
    print("=" * 50)
    healed_analyzer = MapAnalyzer(healed_data)
    print(healed_analyzer.generate_report(args.start))

    return 0


if __name__ == "__main__":
    exit(main())

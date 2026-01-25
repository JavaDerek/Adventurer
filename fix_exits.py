#!/usr/bin/env python3
"""
Fix exits in a rooms JSON file by matching to actual room names.

The problem: exits are often descriptive text like "From the pier down to the shore"
instead of actual room names like "Patriarch's Ponds".

This script:
1. Collects all valid room names
2. For each exit, tries to fuzzy match to actual room names
3. Keeps only valid exits that match real rooms
"""

import argparse
import json
import re
from difflib import SequenceMatcher
from typing import Dict, List, Set, Tuple


def normalize_name(name: str) -> str:
    """Normalize a name for matching."""
    # Remove parenthetical suffixes
    name = re.sub(r'\s*\([^)]*\)\s*', ' ', name)
    # Lowercase and strip
    name = name.lower().strip()
    # Remove extra whitespace
    name = ' '.join(name.split())
    return name


def extract_potential_room_name(exit_text: str) -> List[str]:
    """Extract potential room names from an exit description."""
    candidates = [exit_text]

    # Try to extract room names from various patterns
    patterns = [
        r'^(?:To |Toward |Into |Through |Via |From )(.+?)(?:\s*\(|$)',
        r'^(.+?)(?:\s*\(implied|$)',
        r'^(.+?)(?:\s+via\s+|\s+through\s+)',
        r'(?:labeled |named |called )["\']?([^"\']+)["\']?',
    ]

    for pattern in patterns:
        match = re.search(pattern, exit_text, re.IGNORECASE)
        if match:
            candidates.append(match.group(1).strip())

    return candidates


def find_best_match(exit_text: str, room_names: Set[str], room_names_normalized: Dict[str, str]) -> Tuple[str, float]:
    """Find the best matching room name for an exit."""
    # First, check if exit is already a valid room name
    if exit_text in room_names:
        return exit_text, 1.0

    # Check normalized version
    exit_normalized = normalize_name(exit_text)
    if exit_normalized in room_names_normalized:
        return room_names_normalized[exit_normalized], 0.95

    # Try extracted candidates
    candidates = extract_potential_room_name(exit_text)

    best_match = None
    best_score = 0.0

    for candidate in candidates:
        candidate_normalized = normalize_name(candidate)

        # Check for exact normalized match
        if candidate_normalized in room_names_normalized:
            return room_names_normalized[candidate_normalized], 0.9

        # Check for substring matches (room name contains candidate or vice versa)
        for norm_name, orig_name in room_names_normalized.items():
            if candidate_normalized in norm_name or norm_name in candidate_normalized:
                score = len(min(candidate_normalized, norm_name)) / len(max(candidate_normalized, norm_name))
                if score > best_score and score > 0.5:
                    best_score = score
                    best_match = orig_name

        # Fuzzy matching as last resort
        for norm_name, orig_name in room_names_normalized.items():
            score = SequenceMatcher(None, candidate_normalized, norm_name).ratio()
            if score > best_score and score > 0.7:  # Threshold for fuzzy matching
                best_score = score
                best_match = orig_name

    return best_match, best_score


def fix_exits(rooms_data: dict, min_score: float = 0.7) -> dict:
    """Fix exits in the rooms data."""
    rooms = rooms_data.get("rooms", [])

    # Collect all room names
    room_names = set(room.get("name", "") for room in rooms)
    room_names_normalized = {normalize_name(name): name for name in room_names}

    print(f"Found {len(room_names)} rooms")

    # Track statistics
    total_exits = 0
    matched_exits = 0
    removed_exits = 0

    # Fix each room's exits
    for room in rooms:
        room_name = room.get("name", "")
        old_exits = room.get("exits", [])
        new_exits = []

        for exit_text in old_exits:
            total_exits += 1
            match, score = find_best_match(exit_text, room_names, room_names_normalized)

            if match and score >= min_score:
                # Don't add self-references
                if match != room_name:
                    if match not in new_exits:  # Avoid duplicates
                        new_exits.append(match)
                        matched_exits += 1
            else:
                removed_exits += 1

        room["exits"] = new_exits

    print(f"\nExit Statistics:")
    print(f"  Total exits: {total_exits}")
    print(f"  Matched: {matched_exits}")
    print(f"  Removed: {removed_exits}")

    return rooms_data


def add_bidirectional_connections(rooms_data: dict) -> dict:
    """Add reverse connections where missing."""
    rooms = rooms_data.get("rooms", [])

    # Build room lookup
    room_by_name = {room.get("name"): room for room in rooms}

    connections_added = 0

    for room in rooms:
        room_name = room.get("name", "")
        for exit_name in room.get("exits", []):
            if exit_name in room_by_name:
                target_room = room_by_name[exit_name]
                target_exits = target_room.get("exits", [])
                if room_name not in target_exits:
                    target_exits.append(room_name)
                    target_room["exits"] = target_exits
                    connections_added += 1

    print(f"  Bidirectional connections added: {connections_added}")

    return rooms_data


def connect_disconnected_rooms(rooms_data: dict, start_room: str = None) -> dict:
    """Connect disconnected rooms to the main graph."""
    from analyze_map import MapAnalyzer

    analyzer = MapAnalyzer(rooms_data)
    rooms = rooms_data.get("rooms", [])

    if start_room is None:
        start_room = rooms[0].get("name") if rooms else None

    if not start_room:
        return rooms_data

    # Find disconnected subgraphs
    subgraphs = analyzer.find_disconnected_subgraphs()

    if len(subgraphs) <= 1:
        print("  All rooms already connected")
        return rooms_data

    print(f"  Found {len(subgraphs)} disconnected subgraphs")

    # Find which subgraph contains the start room
    main_subgraph = None
    for sg in subgraphs:
        if start_room in sg:
            main_subgraph = sg
            break

    if not main_subgraph:
        main_subgraph = subgraphs[0]

    # Build room lookup
    room_by_name = {room.get("name"): room for room in rooms}

    # Connect each other subgraph to the main one
    connections_made = 0
    for sg in subgraphs:
        if sg == main_subgraph:
            continue

        # Pick a representative room from each subgraph
        main_room = list(main_subgraph)[0]
        other_room = list(sg)[0]

        # Add bidirectional connection
        if main_room in room_by_name and other_room in room_by_name:
            main_room_data = room_by_name[main_room]
            other_room_data = room_by_name[other_room]

            main_exits = main_room_data.get("exits", [])
            other_exits = other_room_data.get("exits", [])

            if other_room not in main_exits:
                main_exits.append(other_room)
                main_room_data["exits"] = main_exits
                connections_made += 1

            if main_room not in other_exits:
                other_exits.append(main_room)
                other_room_data["exits"] = other_exits
                connections_made += 1

        # Add the newly connected rooms to main_subgraph for next iteration
        main_subgraph = main_subgraph.union(sg)

    print(f"  Cross-subgraph connections added: {connections_made}")

    return rooms_data


def main():
    parser = argparse.ArgumentParser(
        description="Fix exits in a rooms JSON file by matching to actual room names"
    )
    parser.add_argument("json_file", help="Path to the rooms JSON file")
    parser.add_argument("-o", "--output", help="Output file (default: <input>_fixed.json)")
    parser.add_argument("--min-score", type=float, default=0.7,
                        help="Minimum match score (0-1, default: 0.7)")
    parser.add_argument("--no-bidirectional", action="store_true",
                        help="Don't add bidirectional connections")
    parser.add_argument("--connect-subgraphs", action="store_true",
                        help="Connect disconnected subgraphs")
    parser.add_argument("-s", "--start", help="Starting room for analysis")

    args = parser.parse_args()

    # Load input file
    with open(args.json_file, "r") as f:
        rooms_data = json.load(f)

    print("=" * 50)
    print("FIXING EXITS")
    print("=" * 50)

    # Fix exits
    rooms_data = fix_exits(rooms_data, args.min_score)

    # Add bidirectional connections
    if not args.no_bidirectional:
        print("\nAdding bidirectional connections...")
        rooms_data = add_bidirectional_connections(rooms_data)

    # Connect disconnected subgraphs
    if args.connect_subgraphs:
        print("\nConnecting disconnected subgraphs...")
        rooms_data = connect_disconnected_rooms(rooms_data, args.start)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base = args.json_file.rsplit('.', 1)[0]
        output_path = f"{base}_fixed.json"

    # Save
    with open(output_path, "w") as f:
        json.dump(rooms_data, f, indent=2)

    print(f"\nSaved to: {output_path}")

    # Run analysis on result
    print("\n" + "=" * 50)
    print("ANALYSIS OF FIXED MAP")
    print("=" * 50)
    from analyze_map import MapAnalyzer
    analyzer = MapAnalyzer(rooms_data)
    print(analyzer.generate_report(args.start))


if __name__ == "__main__":
    main()

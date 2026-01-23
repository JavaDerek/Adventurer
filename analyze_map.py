#!/usr/bin/env python3
"""
Map Analyzer - Analyzes room connectivity in interactive fiction JSON files.

Identifies broken references, one-way doors, orphaned rooms, and disconnected
subgraphs to help ensure all rooms are properly connected before gameplay.
"""

import json
import argparse
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple


class MapAnalyzer:
    """Analyzes room connectivity in interactive fiction map data."""

    def __init__(self, rooms_data: dict):
        """Initialize with parsed rooms JSON data."""
        self.data = rooms_data
        self.rooms: Dict[str, dict] = {}
        self.graph: Dict[str, List[str]] = defaultdict(list)
        self.reverse_graph: Dict[str, List[str]] = defaultdict(list)
        self._build_structures()

    def _build_structures(self):
        """Build room lookup and graph structures."""
        for room in self.data.get("rooms", []):
            name = room.get("name", "")
            self.rooms[name] = room
            for exit_name in room.get("exits", []):
                self.graph[name].append(exit_name)
                self.reverse_graph[exit_name].append(name)

    @classmethod
    def from_file(cls, json_path: str) -> "MapAnalyzer":
        """Load rooms data from a JSON file."""
        with open(json_path, "r") as f:
            data = json.load(f)
        return cls(data)

    def get_all_room_names(self) -> Set[str]:
        """Get set of all defined room names."""
        return set(self.rooms.keys())

    def find_reachable(self, start_room: str) -> Set[str]:
        """Find all rooms reachable from the starting room using BFS."""
        if start_room not in self.rooms:
            return set()

        visited = set()
        queue = deque([start_room])

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            # Only follow exits to rooms that actually exist
            for exit_name in self.graph.get(current, []):
                if exit_name in self.rooms and exit_name not in visited:
                    queue.append(exit_name)

        return visited

    def find_broken_references(self) -> List[Tuple[str, str]]:
        """Find exits that point to non-existent rooms."""
        broken = []
        room_names = self.get_all_room_names()

        for room_name, exits in self.graph.items():
            for exit_name in exits:
                if exit_name not in room_names:
                    broken.append((room_name, exit_name))

        return broken

    def find_one_way_doors(self) -> List[Tuple[str, str]]:
        """Find connections where A→B exists but B→A doesn't."""
        one_way = []
        room_names = self.get_all_room_names()

        for room_name, exits in self.graph.items():
            for exit_name in exits:
                # Only check if the exit room exists
                if exit_name in room_names:
                    # Check if there's a return path
                    return_exits = self.graph.get(exit_name, [])
                    if room_name not in return_exits:
                        one_way.append((room_name, exit_name))

        return one_way

    def find_unreachable_rooms(self, start_room: str) -> Set[str]:
        """Find rooms that cannot be reached from the starting room."""
        reachable = self.find_reachable(start_room)
        all_rooms = self.get_all_room_names()
        return all_rooms - reachable

    def find_disconnected_subgraphs(self) -> List[Set[str]]:
        """Find groups of rooms that are connected to each other but not to other groups."""
        all_rooms = self.get_all_room_names()
        unvisited = set(all_rooms)
        subgraphs = []

        while unvisited:
            # Pick an arbitrary starting point
            start = next(iter(unvisited))

            # Find all rooms connected to this one (in either direction)
            connected = set()
            queue = deque([start])

            while queue:
                current = queue.popleft()
                if current in connected:
                    continue
                if current not in all_rooms:
                    continue
                connected.add(current)

                # Follow exits in both directions
                for neighbor in self.graph.get(current, []):
                    if neighbor in all_rooms and neighbor not in connected:
                        queue.append(neighbor)
                for neighbor in self.reverse_graph.get(current, []):
                    if neighbor in all_rooms and neighbor not in connected:
                        queue.append(neighbor)

            subgraphs.append(connected)
            unvisited -= connected

        return subgraphs

    def generate_report(self, start_room: str = "TARDIS") -> str:
        """Generate a human-readable connectivity report."""
        lines = []
        lines.append("=" * 50)
        lines.append("MAP ANALYSIS REPORT")
        lines.append("=" * 50)
        lines.append(f"Title: {self.data.get('title', 'Unknown')}")
        lines.append(f"Total rooms: {len(self.rooms)}")
        lines.append("")

        # Reachability
        reachable = self.find_reachable(start_room)
        unreachable = self.find_unreachable_rooms(start_room)
        lines.append(f"Starting room: {start_room}")
        lines.append(f"Reachable rooms: {len(reachable)}")
        lines.append(f"Unreachable rooms: {len(unreachable)}")
        lines.append("")

        # Broken references
        broken = self.find_broken_references()
        lines.append("-" * 50)
        lines.append(f"BROKEN REFERENCES ({len(broken)} found)")
        lines.append("-" * 50)
        if broken:
            for from_room, to_room in broken:
                lines.append(f"  {from_room}")
                lines.append(f"    -> \"{to_room}\" (room doesn't exist)")
        else:
            lines.append("  None - all exits point to valid rooms")
        lines.append("")

        # One-way doors
        one_way = self.find_one_way_doors()
        lines.append("-" * 50)
        lines.append(f"ONE-WAY DOORS ({len(one_way)} found)")
        lines.append("-" * 50)
        if one_way:
            for from_room, to_room in one_way:
                lines.append(f"  {from_room} -> {to_room}")
                lines.append(f"    (no return path from {to_room})")
        else:
            lines.append("  None - all connections are bidirectional")
        lines.append("")

        # Unreachable rooms
        lines.append("-" * 50)
        lines.append(f"UNREACHABLE FROM {start_room} ({len(unreachable)} rooms)")
        lines.append("-" * 50)
        if unreachable:
            for room in sorted(unreachable):
                lines.append(f"  - {room}")
        else:
            lines.append("  None - all rooms are reachable")
        lines.append("")

        # Disconnected subgraphs
        subgraphs = self.find_disconnected_subgraphs()
        lines.append("-" * 50)
        lines.append(f"DISCONNECTED SUBGRAPHS ({len(subgraphs)} found)")
        lines.append("-" * 50)
        if len(subgraphs) > 1:
            for i, subgraph in enumerate(subgraphs, 1):
                # Try to identify the subgraph
                if any("Matrix" in r for r in subgraph):
                    label = "Matrix"
                elif any("TARDIS" in r or "Capitol" in r or "Panopticon" in r for r in subgraph):
                    label = "Capitol"
                else:
                    label = f"Group {i}"
                lines.append(f"  Subgraph {i} ({label}, {len(subgraph)} rooms):")
                for room in sorted(subgraph):
                    lines.append(f"    - {room}")
                lines.append("")
        else:
            lines.append("  All rooms are in one connected graph")

        lines.append("=" * 50)
        return "\n".join(lines)

    def to_json_report(self, start_room: str = "TARDIS") -> dict:
        """Generate a JSON-formatted connectivity report."""
        reachable = self.find_reachable(start_room)
        unreachable = self.find_unreachable_rooms(start_room)
        broken = self.find_broken_references()
        one_way = self.find_one_way_doors()
        subgraphs = self.find_disconnected_subgraphs()

        return {
            "title": self.data.get("title", "Unknown"),
            "total_rooms": len(self.rooms),
            "start_room": start_room,
            "reachable_count": len(reachable),
            "reachable_rooms": sorted(reachable),
            "unreachable_count": len(unreachable),
            "unreachable_rooms": sorted(unreachable),
            "broken_references": [
                {"from": f, "to": t} for f, t in broken
            ],
            "one_way_doors": [
                {"from": f, "to": t} for f, t in one_way
            ],
            "subgraphs": [
                sorted(list(sg)) for sg in subgraphs
            ]
        }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze room connectivity in interactive fiction JSON files"
    )
    parser.add_argument(
        "json_file",
        help="Path to the rooms JSON file"
    )
    parser.add_argument(
        "--start", "-s",
        default="TARDIS",
        help="Starting room for reachability analysis (default: TARDIS)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )

    args = parser.parse_args()

    try:
        analyzer = MapAnalyzer.from_file(args.json_file)
    except FileNotFoundError:
        print(f"Error: File not found: {args.json_file}")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}")
        return 1

    if args.format == "json":
        report = analyzer.to_json_report(args.start)
        print(json.dumps(report, indent=2))
    else:
        print(analyzer.generate_report(args.start))

    return 0


if __name__ == "__main__":
    exit(main())

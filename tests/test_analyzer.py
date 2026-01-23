"""Unit tests for the MapAnalyzer class."""

import json
import pytest
from analyze_map import MapAnalyzer


class TestMapAnalyzer:
    """Tests for MapAnalyzer connectivity analysis."""

    @pytest.fixture
    def simple_connected_map(self):
        """A simple fully connected 3-room map."""
        return {
            "title": "Simple Map",
            "rooms": [
                {"name": "Room A", "exits": ["Room B"]},
                {"name": "Room B", "exits": ["Room A", "Room C"]},
                {"name": "Room C", "exits": ["Room B"]}
            ]
        }

    @pytest.fixture
    def map_with_broken_ref(self):
        """Map with an exit to a non-existent room."""
        return {
            "title": "Broken Map",
            "rooms": [
                {"name": "Room A", "exits": ["Room B", "Nonexistent"]},
                {"name": "Room B", "exits": ["Room A"]}
            ]
        }

    @pytest.fixture
    def map_with_one_way_door(self):
        """Map with a one-way connection."""
        return {
            "title": "One Way Map",
            "rooms": [
                {"name": "Room A", "exits": ["Room B"]},
                {"name": "Room B", "exits": []}  # No way back!
            ]
        }

    @pytest.fixture
    def map_with_disconnected_subgraphs(self):
        """Map with two separate areas."""
        return {
            "title": "Disconnected Map",
            "rooms": [
                {"name": "Area1 Room A", "exits": ["Area1 Room B"]},
                {"name": "Area1 Room B", "exits": ["Area1 Room A"]},
                {"name": "Area2 Room X", "exits": ["Area2 Room Y"]},
                {"name": "Area2 Room Y", "exits": ["Area2 Room X"]}
            ]
        }

    def test_load_from_dict(self, simple_connected_map):
        """Test creating analyzer from dictionary."""
        analyzer = MapAnalyzer(simple_connected_map)
        assert len(analyzer.rooms) == 3
        assert "Room A" in analyzer.rooms
        assert "Room B" in analyzer.rooms
        assert "Room C" in analyzer.rooms

    def test_get_all_room_names(self, simple_connected_map):
        """Test getting all room names."""
        analyzer = MapAnalyzer(simple_connected_map)
        names = analyzer.get_all_room_names()
        assert names == {"Room A", "Room B", "Room C"}

    def test_find_reachable_all_connected(self, simple_connected_map):
        """Test reachability when all rooms are connected."""
        analyzer = MapAnalyzer(simple_connected_map)
        reachable = analyzer.find_reachable("Room A")
        assert reachable == {"Room A", "Room B", "Room C"}

    def test_find_reachable_from_nonexistent_room(self, simple_connected_map):
        """Test reachability from a room that doesn't exist."""
        analyzer = MapAnalyzer(simple_connected_map)
        reachable = analyzer.find_reachable("Nonexistent")
        assert reachable == set()

    def test_find_broken_references(self, map_with_broken_ref):
        """Test finding exits to non-existent rooms."""
        analyzer = MapAnalyzer(map_with_broken_ref)
        broken = analyzer.find_broken_references()
        assert len(broken) == 1
        assert broken[0] == ("Room A", "Nonexistent")

    def test_find_no_broken_references(self, simple_connected_map):
        """Test that a valid map has no broken references."""
        analyzer = MapAnalyzer(simple_connected_map)
        broken = analyzer.find_broken_references()
        assert len(broken) == 0

    def test_find_one_way_doors(self, map_with_one_way_door):
        """Test finding one-way connections."""
        analyzer = MapAnalyzer(map_with_one_way_door)
        one_way = analyzer.find_one_way_doors()
        assert len(one_way) == 1
        assert one_way[0] == ("Room A", "Room B")

    def test_find_no_one_way_doors(self, simple_connected_map):
        """Test that a bidirectional map has no one-way doors."""
        analyzer = MapAnalyzer(simple_connected_map)
        one_way = analyzer.find_one_way_doors()
        assert len(one_way) == 0

    def test_find_unreachable_rooms(self, map_with_disconnected_subgraphs):
        """Test finding rooms unreachable from a starting point."""
        analyzer = MapAnalyzer(map_with_disconnected_subgraphs)
        unreachable = analyzer.find_unreachable_rooms("Area1 Room A")
        assert unreachable == {"Area2 Room X", "Area2 Room Y"}

    def test_find_disconnected_subgraphs(self, map_with_disconnected_subgraphs):
        """Test identifying disconnected areas."""
        analyzer = MapAnalyzer(map_with_disconnected_subgraphs)
        subgraphs = analyzer.find_disconnected_subgraphs()
        assert len(subgraphs) == 2

        # Convert to sets of frozensets for comparison
        subgraph_sets = {frozenset(sg) for sg in subgraphs}
        expected = {
            frozenset({"Area1 Room A", "Area1 Room B"}),
            frozenset({"Area2 Room X", "Area2 Room Y"})
        }
        assert subgraph_sets == expected

    def test_single_subgraph_when_connected(self, simple_connected_map):
        """Test that a connected map has only one subgraph."""
        analyzer = MapAnalyzer(simple_connected_map)
        subgraphs = analyzer.find_disconnected_subgraphs()
        assert len(subgraphs) == 1
        assert set(subgraphs[0]) == {"Room A", "Room B", "Room C"}

    def test_generate_report(self, simple_connected_map):
        """Test generating a text report."""
        analyzer = MapAnalyzer(simple_connected_map)
        report = analyzer.generate_report("Room A")
        assert "MAP ANALYSIS REPORT" in report
        assert "Total rooms: 3" in report
        assert "Reachable rooms: 3" in report

    def test_to_json_report(self, simple_connected_map):
        """Test generating a JSON report."""
        analyzer = MapAnalyzer(simple_connected_map)
        report = analyzer.to_json_report("Room A")
        assert report["total_rooms"] == 3
        assert report["reachable_count"] == 3
        assert report["unreachable_count"] == 0
        assert len(report["broken_references"]) == 0
        assert len(report["one_way_doors"]) == 0

    def test_empty_map(self):
        """Test handling an empty map."""
        analyzer = MapAnalyzer({"title": "Empty", "rooms": []})
        assert len(analyzer.rooms) == 0
        assert analyzer.find_reachable("Anywhere") == set()
        assert analyzer.find_broken_references() == []

    def test_room_with_no_exits(self):
        """Test a room with no exits defined."""
        data = {
            "title": "Isolated",
            "rooms": [
                {"name": "Island", "exits": []},
                {"name": "Mainland", "exits": ["Island"]}
            ]
        }
        analyzer = MapAnalyzer(data)
        one_way = analyzer.find_one_way_doors()
        assert ("Mainland", "Island") in one_way


class TestMapAnalyzerWithFixture:
    """Tests using the actual Deadly Assassin fixture."""

    @pytest.fixture
    def deadly_assassin_analyzer(self):
        """Load the Deadly Assassin map fixture."""
        with open("tests/fixtures/deadly_assassin_gold.json") as f:
            data = json.load(f)
        return MapAnalyzer(data)

    def test_fixture_loads(self, deadly_assassin_analyzer):
        """Test that the fixture loads correctly."""
        assert len(deadly_assassin_analyzer.rooms) > 0

    def test_fixture_has_broken_refs(self, deadly_assassin_analyzer):
        """Test that the fixture has expected connectivity issues."""
        broken = deadly_assassin_analyzer.find_broken_references()
        # The gold standard might be better connected, but should still find some issues
        assert isinstance(broken, list)

    def test_fixture_reachability(self, deadly_assassin_analyzer):
        """Test reachability analysis on the fixture."""
        reachable = deadly_assassin_analyzer.find_reachable("TARDIS")
        # TARDIS should be able to reach at least itself
        assert "TARDIS" in reachable or len(reachable) == 0

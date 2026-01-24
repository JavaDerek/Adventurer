"""Tests for load_to_dmcp.py"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

# Import the functions we want to test
import sys
sys.path.insert(0, '.')
from load_to_dmcp import (
    normalize_character_name,
    extract_id_from_result,
    load_game_to_dmcp,
    load_game_with_session,
)


class TestNormalizeCharacterName:
    """Tests for character name normalization."""

    def test_simple_name(self):
        """Simple names should be unchanged."""
        assert normalize_character_name("Doctor") == "Doctor"

    def test_name_with_parenthetical(self):
        """Parentheticals should be removed."""
        assert normalize_character_name("Hildred (briefly)") == "Hildred"

    def test_name_with_via_parenthetical(self):
        """Via parentheticals should be removed."""
        assert normalize_character_name("Spandrell (via viewscreen)") == "Spandrell"

    def test_name_with_complex_parenthetical(self):
        """Complex parentheticals should be removed."""
        assert normalize_character_name("Master (as surgeon/voice)") == "Master"

    def test_name_with_multiple_parentheticals(self):
        """Multiple parentheticals should all be removed."""
        assert normalize_character_name("Guard (shot) (dead)") == "Guard"

    def test_name_with_extra_whitespace(self):
        """Extra whitespace should be normalized."""
        assert normalize_character_name("  Doctor   Who  ") == "Doctor Who"

    def test_name_with_parenthetical_and_whitespace(self):
        """Parentheticals and whitespace should both be handled."""
        assert normalize_character_name("  Hildred  (briefly)  ") == "Hildred"


class TestExtractIdFromResult:
    """Tests for extracting IDs from MCP tool results."""

    def test_extract_id_from_json_content(self):
        """Should extract ID from JSON in content."""
        result = MagicMock()
        result.content = [MagicMock(text='{"id": "abc-123", "name": "Test"}')]

        assert extract_id_from_result(result, "game") == "abc-123"

    def test_extract_game_id_field(self):
        """Should extract gameId field."""
        result = MagicMock()
        result.content = [MagicMock(text='{"gameId": "game-456"}')]

        assert extract_id_from_result(result, "game") == "game-456"

    def test_extract_location_id_field(self):
        """Should extract locationId field."""
        result = MagicMock()
        result.content = [MagicMock(text='{"locationId": "loc-789"}')]

        assert extract_id_from_result(result, "location") == "loc-789"

    def test_extract_nested_id(self):
        """Should extract ID from nested object."""
        result = MagicMock()
        result.content = [MagicMock(text='{"game": {"id": "nested-123"}}')]

        assert extract_id_from_result(result, "game") == "nested-123"

    def test_extract_id_via_regex_fallback(self):
        """Should fall back to regex extraction."""
        result = MagicMock()
        result.content = [MagicMock(text='Some text with "id": "regex-456" in it')]

        assert extract_id_from_result(result, "game") == "regex-456"

    def test_extract_id_no_content_attribute(self):
        """Should handle results without content attribute."""
        result = "raw-string-result"

        assert extract_id_from_result(result, "game") == "raw-string-result"

    def test_extract_id_empty_content(self):
        """Should handle empty content list."""
        result = MagicMock()
        result.content = []

        # Should return string representation
        assert "MagicMock" in extract_id_from_result(result, "game")


class TestLoaderWithFixture:
    """Tests using the gold standard fixture."""

    @pytest.fixture
    def gold_data(self):
        """Load the gold standard fixture."""
        with open("tests/fixtures/deadly_assassin_gold.json") as f:
            return json.load(f)

    def test_fixture_has_expected_rooms(self, gold_data):
        """Fixture should have 28 rooms."""
        assert len(gold_data["rooms"]) == 28

    def test_all_rooms_have_required_fields(self, gold_data):
        """All rooms should have name, description, exits."""
        for room in gold_data["rooms"]:
            assert "name" in room
            assert "description" in room
            assert "exits" in room

    def test_character_deduplication(self, gold_data):
        """Character deduplication should reduce count."""
        all_chars = []
        for room in gold_data["rooms"]:
            all_chars.extend(room.get("characters", []))

        # Normalize and dedupe
        unique_chars = set(normalize_character_name(c) for c in all_chars)

        # Should have fewer unique than total
        assert len(unique_chars) < len(all_chars)
        # Should have exactly 25 unique characters
        assert len(unique_chars) == 25

    def test_exits_reference_existing_rooms(self, gold_data):
        """Count how many exits reference existing rooms."""
        room_names = {room["name"] for room in gold_data["rooms"]}

        valid_exits = 0
        invalid_exits = 0

        for room in gold_data["rooms"]:
            for exit_name in room.get("exits", []):
                if exit_name in room_names:
                    valid_exits += 1
                else:
                    invalid_exits += 1

        # Should have some valid exits
        assert valid_exits > 0
        # Document the mismatch (exit names don't match room names exactly)
        assert invalid_exits > 0  # This is expected given the data


class TestLoadGameWithSession:
    """Tests for load_game_with_session using mocked MCP session."""

    @pytest.fixture
    def simple_game_data(self):
        """Simple game data for testing."""
        return {
            "title": "Test Adventure",
            "rooms": [
                {
                    "name": "Start Room",
                    "description": "The starting room.",
                    "exits": ["End Room"],
                    "items": ["key"],
                    "characters": ["Player"],
                    "events": ["Game begins"],
                    "atmosphere": "calm"
                }
            ]
        }

    @pytest.fixture
    def two_room_game_data(self):
        """Game data with two connected rooms."""
        return {
            "title": "Two Room Adventure",
            "rooms": [
                {
                    "name": "Start Room",
                    "description": "The starting room.",
                    "exits": ["End Room"],
                    "items": ["key"],
                    "characters": ["Hero"],
                    "events": ["Adventure begins"],
                    "atmosphere": "peaceful"
                },
                {
                    "name": "End Room",
                    "description": "The final room.",
                    "exits": ["Start Room"],
                    "items": ["treasure"],
                    "characters": ["Dragon"],
                    "events": ["Boss appears"],
                    "atmosphere": "tense"
                }
            ]
        }

    def _make_mock_result(self, id_value):
        """Create a mock MCP result with an ID."""
        result = MagicMock()
        result.content = [MagicMock(text=json.dumps({"id": id_value}))]
        return result

    @pytest.mark.asyncio
    async def test_creates_game_with_title(self, simple_game_data):
        """Should create game with the correct title."""
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=self._make_mock_result("game-123"))

        await load_game_with_session(mock_session, simple_game_data)

        # First call should be create_game
        calls = mock_session.call_tool.call_args_list
        assert calls[0][0][0] == "create_game"
        assert calls[0][0][1]["name"] == "Test Adventure"

    @pytest.mark.asyncio
    async def test_creates_locations_for_each_room(self, two_room_game_data):
        """Should create a location for each room."""
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=self._make_mock_result("id-123"))

        await load_game_with_session(mock_session, two_room_game_data)

        # Find all create_location calls
        calls = mock_session.call_tool.call_args_list
        location_calls = [c for c in calls if c[0][0] == "create_location"]
        assert len(location_calls) == 2

    @pytest.mark.asyncio
    async def test_connects_locations_based_on_exits(self, two_room_game_data):
        """Should connect locations based on exit definitions."""
        mock_session = AsyncMock()
        # Return different IDs for game, locations
        mock_session.call_tool = AsyncMock(side_effect=[
            self._make_mock_result("game-1"),      # create_game
            self._make_mock_result("loc-start"),   # create_location (Start Room)
            self._make_mock_result("loc-end"),     # create_location (End Room)
            self._make_mock_result("conn-1"),      # connect_locations
            self._make_mock_result("conn-2"),      # connect_locations
            self._make_mock_result("char-1"),      # create_character (Hero)
            self._make_mock_result("char-2"),      # create_character (Dragon)
            self._make_mock_result("item-1"),      # create_item (key)
            self._make_mock_result("item-2"),      # create_item (treasure)
            self._make_mock_result("event-1"),     # add_event
            self._make_mock_result("event-2"),     # add_event
        ])

        await load_game_with_session(mock_session, two_room_game_data)

        # Find all connect_locations calls
        calls = mock_session.call_tool.call_args_list
        connect_calls = [c for c in calls if c[0][0] == "connect_locations"]
        assert len(connect_calls) == 2

    @pytest.mark.asyncio
    async def test_creates_characters_with_deduplication(self):
        """Should deduplicate characters with parentheticals."""
        game_data = {
            "title": "Character Test",
            "rooms": [
                {
                    "name": "Room A",
                    "description": "First room",
                    "exits": [],
                    "characters": ["Hildred", "Hildred (briefly)", "Spandrell (via viewscreen)"]
                },
                {
                    "name": "Room B",
                    "description": "Second room",
                    "exits": [],
                    "characters": ["Hildred", "Spandrell"]
                }
            ]
        }

        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=self._make_mock_result("id-123"))

        result = await load_game_with_session(mock_session, game_data)

        # Should only have 2 unique characters: Hildred and Spandrell
        assert len(result["character_ids"]) == 2
        assert "Hildred" in result["character_ids"]
        assert "Spandrell" in result["character_ids"]

    @pytest.mark.asyncio
    async def test_creates_items_at_locations(self, simple_game_data):
        """Should create items at their locations."""
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=self._make_mock_result("id-123"))

        result = await load_game_with_session(mock_session, simple_game_data)

        # Find create_item calls
        calls = mock_session.call_tool.call_args_list
        item_calls = [c for c in calls if c[0][0] == "create_item"]
        assert len(item_calls) == 1
        assert result["items_created"] == 1

    @pytest.mark.asyncio
    async def test_adds_events(self, simple_game_data):
        """Should add events from room data."""
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=self._make_mock_result("id-123"))

        result = await load_game_with_session(mock_session, simple_game_data)

        assert result["events_added"] == 1

    @pytest.mark.asyncio
    async def test_appends_atmosphere_to_description(self, simple_game_data):
        """Should append atmosphere to location description."""
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=self._make_mock_result("id-123"))

        await load_game_with_session(mock_session, simple_game_data)

        # Find create_location call
        calls = mock_session.call_tool.call_args_list
        location_call = next(c for c in calls if c[0][0] == "create_location")
        description = location_call[0][1]["description"]
        assert "Atmosphere: calm" in description

    @pytest.mark.asyncio
    async def test_returns_summary_dict(self, simple_game_data):
        """Should return summary with all IDs and counts."""
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=self._make_mock_result("test-id"))

        result = await load_game_with_session(mock_session, simple_game_data)

        assert "game_id" in result
        assert "title" in result
        assert "location_ids" in result
        assert "character_ids" in result
        assert "items_created" in result
        assert "events_added" in result
        assert result["title"] == "Test Adventure"

    @pytest.mark.asyncio
    async def test_handles_missing_optional_fields(self):
        """Should handle rooms missing optional fields."""
        game_data = {
            "title": "Minimal Game",
            "rooms": [
                {
                    "name": "Empty Room",
                    "description": "Nothing here.",
                    "exits": []
                    # No items, characters, events, atmosphere
                }
            ]
        }

        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=self._make_mock_result("id-123"))

        result = await load_game_with_session(mock_session, game_data)

        assert result["items_created"] == 0
        assert result["events_added"] == 0
        assert len(result["character_ids"]) == 0

    @pytest.mark.asyncio
    async def test_handles_item_creation_failure(self):
        """Should continue if item creation fails."""
        game_data = {
            "title": "Item Test",
            "rooms": [
                {
                    "name": "Room",
                    "description": "A room",
                    "exits": [],
                    "items": ["item1", "item2", "item3"]
                }
            ]
        }

        mock_session = AsyncMock()
        # Fail on second item
        mock_session.call_tool = AsyncMock(side_effect=[
            self._make_mock_result("game-1"),    # create_game
            self._make_mock_result("loc-1"),     # create_location
            self._make_mock_result("item-1"),    # create_item (succeeds)
            Exception("Item creation failed"),   # create_item (fails)
            self._make_mock_result("item-3"),    # create_item (succeeds)
        ])

        result = await load_game_with_session(mock_session, game_data)

        # Should have created 2 items despite one failure
        assert result["items_created"] == 2

    @pytest.mark.asyncio
    async def test_uses_custom_setting_and_style(self, simple_game_data):
        """Should pass custom setting and style to create_game."""
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=self._make_mock_result("game-123"))

        await load_game_with_session(
            mock_session,
            simple_game_data,
            setting="sci-fi space opera",
            style="action and exploration"
        )

        calls = mock_session.call_tool.call_args_list
        create_game_call = calls[0]
        assert create_game_call[0][1]["setting"] == "sci-fi space opera"
        assert create_game_call[0][1]["style"] == "action and exploration"


class TestLoadGameToDMCP:
    """Tests for the main load_game_to_dmcp function."""

    @pytest.fixture
    def simple_game_data(self):
        """Simple game data for testing."""
        return {
            "title": "Test Adventure",
            "rooms": [
                {
                    "name": "Start Room",
                    "description": "The starting room.",
                    "exits": ["End Room"],
                    "items": ["key"],
                    "characters": ["Player"],
                    "events": ["Game begins"],
                    "atmosphere": "calm"
                }
            ]
        }

    @pytest.mark.asyncio
    async def test_load_game_handles_missing_dmcp(self, simple_game_data, tmp_path):
        """Should exit gracefully if DMCP is not found."""
        json_file = tmp_path / "test_game.json"
        json_file.write_text(json.dumps(simple_game_data))

        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(SystemExit) as exc_info:
                await load_game_to_dmcp(
                    json_path=str(json_file),
                    dmcp_path="/nonexistent/dmcp"
                )
            assert exc_info.value.code == 1

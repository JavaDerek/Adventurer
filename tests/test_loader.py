"""Tests for load_to_run_dmcp.py"""

import asyncio
import json
import logging

# Import the functions we want to test
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, '.')
from load_to_run_dmcp import (
    DESCRIPTION_MAX,
    NAME_MAX,
    _disable_structured_output_validation,
    call_tool,
    exit_direction,
    extract_id_from_result,
    extract_web_ui_url,
    is_schema_drift_error,
    load_game_to_run_dmcp,
    load_game_with_session,
    normalize_character_name,
    reset_schema_drift_state,
    schema_drift_detected,
    truncate,
)


class TestNormalizeCharacterName:
    """Tests for character name normalization."""

    def test_simple_name(self):
        """Simple names should be unchanged."""
        assert normalize_character_name("Raskolnikov") == "Raskolnikov"

    def test_name_with_parenthetical(self):
        """Parentheticals should be removed."""
        assert normalize_character_name("Sonia (briefly)") == "Sonia"

    def test_name_with_via_parenthetical(self):
        """Via parentheticals should be removed."""
        assert normalize_character_name("Porfiry (via the doorway)") == "Porfiry"

    def test_name_with_complex_parenthetical(self):
        """Complex parentheticals should be removed."""
        assert normalize_character_name("Svidrigailov (as ghost/voice)") == "Svidrigailov"

    def test_name_with_multiple_parentheticals(self):
        """Multiple parentheticals should all be removed."""
        assert normalize_character_name("Guard (shot) (dead)") == "Guard"

    def test_name_with_extra_whitespace(self):
        """Extra whitespace should be normalized."""
        assert normalize_character_name("  Katerina   Ivanovna  ") == "Katerina Ivanovna"

    def test_name_with_parenthetical_and_whitespace(self):
        """Parentheticals and whitespace should both be handled."""
        assert normalize_character_name("  Sonia  (briefly)  ") == "Sonia"


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
    """Tests using the gold standard fixture.

    The fixture is a raw extraction from the public-domain Constance Garnett
    translation of Crime and Punishment -- pre-fix_exits.py, so it still has
    the descriptive exits a fresh LLM run produces.
    """

    @pytest.fixture
    def gold_data(self):
        """Load the gold standard fixture."""
        with open("tests/fixtures/crime_and_punishment_gold.json") as f:
            return json.load(f)

    def test_fixture_has_expected_rooms(self, gold_data):
        """Fixture should have 90 rooms."""
        assert len(gold_data["rooms"]) == 90

    def test_fixture_is_public_domain(self, gold_data):
        """The committed fixture must stay redistributable."""
        assert gold_data["source"] == "public_domain_novel"
        assert "Garnett" in gold_data["provenance"]["translation"]

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
        unique_chars = {normalize_character_name(c) for c in all_chars}

        # Should have fewer unique than total
        assert len(unique_chars) < len(all_chars)
        # Should have exactly 113 unique characters
        assert len(unique_chars) == 113

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
            self._make_mock_result("note-1"),      # create_note (beats)
            self._make_mock_result("note-2"),      # create_note (beats)
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
                    "characters": ["Sonia", "Sonia (briefly)", "Porfiry (via the doorway)"]
                },
                {
                    "name": "Room B",
                    "description": "Second room",
                    "exits": [],
                    "characters": ["Sonia", "Porfiry"]
                }
            ]
        }

        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=self._make_mock_result("id-123"))

        result = await load_game_with_session(mock_session, game_data)

        # Should only have 2 unique characters: Sonia and Porfiry
        assert len(result["character_ids"]) == 2
        assert "Sonia" in result["character_ids"]
        assert "Porfiry" in result["character_ids"]

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
    async def test_records_beats_as_notes(self, simple_game_data):
        """Source beats become GM notes, not played history."""
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=self._make_mock_result("id-123"))

        result = await load_game_with_session(mock_session, simple_game_data)

        assert result["notes_created"] == 1
        assert result["beats_recorded"] == 1

        calls = mock_session.call_tool.call_args_list
        # Beats must NOT be written into narrative history.
        assert not [c for c in calls if c[0][0] in ("add_event", "log_event")]

        note_call = next(c for c in calls if c[0][0] == "create_note")
        args = note_call[0][1]
        assert args["category"] == "plot"
        assert args["relatedEntityType"] == "location"
        assert args["relatedEntityId"] == "id-123"
        assert "Game begins" in args["content"]
        assert "Start Room" in args["title"]

    @pytest.mark.asyncio
    async def test_one_note_per_room_not_per_beat(self):
        """A room with three beats produces one note holding all three."""
        game_data = {
            "title": "Beats",
            "rooms": [{
                "name": "Room",
                "description": "A room.",
                "exits": [],
                "events": ["First", "Second", "Third"],
            }],
        }
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=self._make_mock_result("id-123"))

        result = await load_game_with_session(mock_session, game_data)

        calls = mock_session.call_tool.call_args_list
        note_calls = [c for c in calls if c[0][0] == "create_note"]
        assert len(note_calls) == 1
        assert result["notes_created"] == 1
        assert result["beats_recorded"] == 3
        content = note_calls[0][0][1]["content"]
        for beat in ("First", "Second", "Third"):
            assert beat in content

    @pytest.mark.asyncio
    async def test_atmosphere_uses_structured_property(self, simple_game_data):
        """Atmosphere belongs in properties.atmosphere, which run-dmcp stores."""
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=self._make_mock_result("id-123"))

        await load_game_with_session(mock_session, simple_game_data)

        calls = mock_session.call_tool.call_args_list
        location_call = next(c for c in calls if c[0][0] == "create_location")
        args = location_call[0][1]
        assert args["properties"]["atmosphere"] == "calm"
        # The description stays clean prose -- no "Atmosphere:" suffix jammed in.
        assert "Atmosphere:" not in args["description"]

    @pytest.mark.asyncio
    async def test_omits_properties_when_no_atmosphere(self):
        """Rooms without atmosphere should not send an empty properties block."""
        game_data = {
            "title": "No Atmosphere",
            "rooms": [{"name": "Room", "description": "A room.", "exits": []}],
        }
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=self._make_mock_result("id-123"))

        await load_game_with_session(mock_session, game_data)

        calls = mock_session.call_tool.call_args_list
        location_call = next(c for c in calls if c[0][0] == "create_location")
        assert "properties" not in location_call[0][1]

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
        assert "notes_created" in result
        assert "beats_recorded" in result
        assert "connections_made" in result
        assert "web_ui_url" in result
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
        assert result["notes_created"] == 0
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


class TestLoadGameToRunDmcp:
    """Tests for the main load_game_to_run_dmcp function."""

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
    async def test_load_game_handles_missing_server(self, simple_game_data, tmp_path):
        """Should exit gracefully if run-dmcp is not found."""
        json_file = tmp_path / "test_game.json"
        json_file.write_text(json.dumps(simple_game_data))

        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(SystemExit) as exc_info:
                await load_game_to_run_dmcp(
                    json_path=str(json_file),
                    server_path="/nonexistent/run-dmcp"
                )
            assert exc_info.value.code == 1


class TestExitDirection:
    """run-dmcp keys exits by direction, so directions must be unique per room."""

    def test_direction_names_the_destination(self):
        assert exit_direction("Haymarket") == "toward Haymarket"

    def test_directions_differ_per_destination(self):
        """Two exits from one room must not collide on direction."""
        # connectLocations() drops any existing exit sharing a direction, so a
        # constant string here would leave every room with exactly one exit.
        assert exit_direction("Haymarket") != exit_direction("Police Station")

    def test_direction_respects_name_limit(self):
        long_name = "x" * 400
        assert len(exit_direction(long_name)) <= NAME_MAX


class TestTruncate:
    """Oversized fields are clamped, not sent to be rejected."""

    def test_short_text_untouched(self):
        assert truncate("hello", 200) == "hello"

    def test_exact_length_untouched(self):
        assert truncate("abcde", 5) == "abcde"

    def test_long_text_clamped_to_limit(self):
        assert len(truncate("x" * 500, 200)) == 200

    def test_handles_none(self):
        assert truncate(None, 200) == ""


class TestExtractWebUiUrl:
    """run-dmcp ships a web UI and returns its URL from create_game."""

    def test_extracts_url(self):
        result = MagicMock()
        result.content = [MagicMock(text=json.dumps(
            {"id": "g1", "webUi": {"url": "http://localhost:3456/games/g1"}}
        ))]
        assert extract_web_ui_url(result) == "http://localhost:3456/games/g1"

    def test_returns_none_when_absent(self):
        result = MagicMock()
        result.content = [MagicMock(text=json.dumps({"id": "g1"}))]
        assert extract_web_ui_url(result) is None

    def test_returns_none_on_unparseable(self):
        result = MagicMock()
        result.content = [MagicMock(text="not json")]
        assert extract_web_ui_url(result) is None


class TestRunDmcpToolContract:
    """The call shapes run-dmcp actually accepts."""

    def _make_mock_result(self, id_value):
        result = MagicMock()
        result.content = [MagicMock(text=json.dumps({"id": id_value}))]
        return result

    @pytest.fixture
    def two_room_game_data(self):
        return {
            "title": "Two Room Adventure",
            "rooms": [
                {"name": "Start Room", "description": "First.", "exits": ["End Room"]},
                {"name": "End Room", "description": "Last.", "exits": ["Start Room"]},
            ],
        }

    @pytest.mark.asyncio
    async def test_connect_uses_run_dmcp_parameter_names(self, two_room_game_data):
        """fromId/toId were DMCP's names; run-dmcp wants from/toLocationId."""
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(side_effect=[
            self._make_mock_result("game-1"),
            self._make_mock_result("loc-start"),
            self._make_mock_result("loc-end"),
            self._make_mock_result("conn-1"),
            self._make_mock_result("conn-2"),
        ])

        await load_game_with_session(mock_session, two_room_game_data)

        calls = mock_session.call_tool.call_args_list
        connect = next(c for c in calls if c[0][0] == "connect_locations")[0][1]
        assert connect["fromLocationId"] == "loc-start"
        assert connect["toLocationId"] == "loc-end"
        assert connect["fromDirection"] == "toward End Room"
        assert connect["toDirection"] == "toward Start Room"
        # No gameId on this tool in run-dmcp.
        assert "gameId" not in connect
        assert "fromId" not in connect

    @pytest.mark.asyncio
    async def test_connect_is_not_bidirectional(self, two_room_game_data):
        """bidirectional defaults to true upstream; the JSON already lists both sides."""
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=self._make_mock_result("id-1"))

        await load_game_with_session(mock_session, two_room_game_data)

        calls = mock_session.call_tool.call_args_list
        connects = [c[0][1] for c in calls if c[0][0] == "connect_locations"]
        assert connects, "expected connections"
        for connect in connects:
            assert connect["bidirectional"] is False

    @pytest.mark.asyncio
    async def test_one_way_exit_stays_one_way(self):
        """A room list with a single direction must not gain a reverse exit."""
        game_data = {
            "title": "One Way",
            "rooms": [
                {"name": "A", "description": "a", "exits": ["B"]},
                {"name": "B", "description": "b", "exits": []},
            ],
        }
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=self._make_mock_result("id-1"))

        result = await load_game_with_session(mock_session, game_data)

        calls = mock_session.call_tool.call_args_list
        connects = [c[0][1] for c in calls if c[0][0] == "connect_locations"]
        assert len(connects) == 1
        assert connects[0]["bidirectional"] is False
        assert result["connections_made"] == 1

    @pytest.mark.asyncio
    async def test_duplicate_exits_deduped(self):
        """The same destination listed twice must not emit two connections."""
        game_data = {
            "title": "Dupes",
            "rooms": [
                {"name": "A", "description": "a", "exits": ["B", "B", "B"]},
                {"name": "B", "description": "b", "exits": []},
            ],
        }
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=self._make_mock_result("id-1"))

        result = await load_game_with_session(mock_session, game_data)

        calls = mock_session.call_tool.call_args_list
        connects = [c for c in calls if c[0][0] == "connect_locations"]
        assert len(connects) == 1
        assert result["connections_made"] == 1

    @pytest.mark.asyncio
    async def test_self_referential_exit_skipped(self):
        """A room listing itself as an exit would collide with its own direction."""
        game_data = {
            "title": "Self",
            "rooms": [{"name": "A", "description": "a", "exits": ["A"]}],
        }
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=self._make_mock_result("id-1"))

        result = await load_game_with_session(mock_session, game_data)

        calls = mock_session.call_tool.call_args_list
        assert not [c for c in calls if c[0][0] == "connect_locations"]
        assert result["connections_made"] == 0

    @pytest.mark.asyncio
    async def test_create_item_uses_owner_fields(self):
        """DMCP took locationId; run-dmcp takes ownerId + ownerType."""
        game_data = {
            "title": "Items",
            "rooms": [{"name": "Room", "description": "r", "exits": [], "items": ["key"]}],
        }
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=self._make_mock_result("loc-1"))

        await load_game_with_session(mock_session, game_data)

        calls = mock_session.call_tool.call_args_list
        item = next(c for c in calls if c[0][0] == "create_item")[0][1]
        assert item["ownerId"] == "loc-1"
        assert item["ownerType"] == "location"
        assert item["name"] == "key"
        assert item["properties"]["description"]
        assert "locationId" not in item

    @pytest.mark.asyncio
    async def test_oversized_fields_are_clamped(self):
        """A 568-room load must not abort on one overlong field."""
        game_data = {
            "title": "T" * 400,
            "rooms": [{
                "name": "N" * 400,
                "description": "D" * 9000,
                "exits": [],
            }],
        }
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=self._make_mock_result("id-1"))

        await load_game_with_session(mock_session, game_data)

        calls = mock_session.call_tool.call_args_list
        assert len(calls[0][0][1]["name"]) <= NAME_MAX
        location = next(c for c in calls if c[0][0] == "create_location")[0][1]
        assert len(location["name"]) <= NAME_MAX
        assert len(location["description"]) <= DESCRIPTION_MAX

    @pytest.mark.asyncio
    async def test_reports_web_ui_url(self):
        """The loader should hand back run-dmcp's web UI link."""
        game_data = {"title": "T", "rooms": [{"name": "R", "description": "r", "exits": []}]}
        result_with_url = MagicMock()
        result_with_url.content = [MagicMock(text=json.dumps(
            {"id": "game-9", "webUi": {"url": "http://localhost:3456/games/game-9"}}
        ))]
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(side_effect=[
            result_with_url,
            self._make_mock_result("loc-1"),
        ])

        result = await load_game_with_session(mock_session, game_data)

        assert result["web_ui_url"] == "http://localhost:3456/games/game-9"

    @pytest.mark.asyncio
    async def test_player_character_flagged(self):
        """A character literally named Player is the PC."""
        game_data = {
            "title": "PC",
            "rooms": [{
                "name": "R", "description": "r", "exits": [],
                "characters": ["Player", "Guard"],
            }],
        }
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=self._make_mock_result("id-1"))

        await load_game_with_session(mock_session, game_data)

        calls = mock_session.call_tool.call_args_list
        chars = {c[0][1]["name"]: c[0][1] for c in calls if c[0][0] == "create_character"}
        assert chars["Player"]["isPlayer"] is True
        assert chars["Guard"]["isPlayer"] is False


class TestStructuredOutputValidation:
    """run-dmcp's create_character outputSchema omits fields it returns."""

    def test_patches_public_hook(self):
        class FakeSession:
            async def validate_tool_result(self, name, result):
                raise RuntimeError("should not run")

        patched = _disable_structured_output_validation(FakeSession)

        assert "validate_tool_result" in patched

    def test_patches_legacy_private_hook(self):
        class LegacySession:
            async def _validate_tool_result(self, name, result):
                raise RuntimeError("should not run")

        patched = _disable_structured_output_validation(LegacySession)

        assert "_validate_tool_result" in patched

    def test_patched_hook_is_a_noop(self):
        class FakeSession:
            async def validate_tool_result(self, name, result):
                raise RuntimeError("should not run")

        _disable_structured_output_validation(FakeSession)

        # The replacement must not raise on a payload that fails the schema.
        asyncio.run(FakeSession().validate_tool_result("create_character", object()))

    def test_warns_when_no_hook_exists(self, caplog):
        """A future SDK rename must be loud, not silent."""
        class UnknownSession:
            pass

        with caplog.at_level(logging.WARNING):
            patched = _disable_structured_output_validation(UnknownSession)

        assert patched == []
        assert "output-validation hook" in caplog.text


class TestServerEnvironment:
    """The SDK scrubs the child environment unless we pass one."""

    @pytest.mark.asyncio
    async def test_server_params_forward_environment(self, tmp_path, monkeypatch):
        """DMCP_DB_PATH must reach the server, or it writes to the wrong database."""
        server_root = tmp_path / "run-dmcp"
        (server_root / "dist").mkdir(parents=True)
        (server_root / "dist" / "index.js").write_text("// stub")

        game_file = tmp_path / "game.json"
        game_file.write_text(json.dumps({"title": "T", "rooms": []}))

        monkeypatch.setenv("DMCP_DB_PATH", "/tmp/scratch-games.db")

        captured = {}

        class FakeStdioClient:
            def __init__(self, params):
                captured["params"] = params

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
                return None

        with patch("mcp.client.stdio.stdio_client", FakeStdioClient), \
             patch("mcp.ClientSession", FakeSession), \
             patch("load_to_run_dmcp.load_game_with_session",
                   AsyncMock(return_value={"game_id": "g"})):
            await load_game_to_run_dmcp(
                json_path=str(game_file),
                server_path=str(server_root),
            )

        env = captured["params"].env
        assert env is not None, "StdioServerParameters.env must be set"
        assert env["DMCP_DB_PATH"] == "/tmp/scratch-games.db"


class TestSchemaDriftTolerance:
    """Validation stays on; it degrades only if a server actually violates it.

    run-dmcp#24 is fixed upstream, so a current server passes validation and
    real schema drift should surface. A stale checkout must still load, with an
    explanation rather than a silent 'Characters: 0'.
    """

    def setup_method(self):
        reset_schema_drift_state()

    def teardown_method(self):
        reset_schema_drift_state()

    def test_recognises_schema_drift_error(self):
        exc = RuntimeError(
            "Invalid structured content returned by tool create_character: "
            "Additional properties are not allowed ('imageGen', 'voice' were unexpected)"
        )
        assert is_schema_drift_error(exc)

    def test_unrelated_runtime_error_is_not_drift(self):
        assert not is_schema_drift_error(RuntimeError("connection reset"))

    @pytest.mark.asyncio
    async def test_validation_left_alone_on_a_healthy_server(self):
        """No drift means no patching -- real errors keep surfacing."""
        session = AsyncMock()
        session.call_tool = AsyncMock(return_value="ok")

        result = await call_tool(session, "create_character", {"name": "x"})

        assert result == "ok"
        assert session.call_tool.await_count == 1
        assert schema_drift_detected() is False

    @pytest.mark.asyncio
    async def test_non_drift_errors_propagate(self):
        """A genuine failure must not be swallowed by the drift handler."""
        session = AsyncMock()
        session.call_tool = AsyncMock(side_effect=RuntimeError("connection reset"))

        with pytest.raises(RuntimeError, match="connection reset"):
            await call_tool(session, "create_character", {})

        assert schema_drift_detected() is False

    @pytest.mark.asyncio
    async def test_drift_degrades_once_then_retries(self, caplog):
        """First drift: warn, disable validation, retry. Then carry on."""
        class FakeSession:
            def __init__(self):
                self.calls = 0

            async def validate_tool_result(self, name, result):
                return None

            async def call_tool(self, name, arguments):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError(
                        "Invalid structured content returned by tool "
                        "create_character: Additional properties are not "
                        "allowed ('imageGen', 'voice' were unexpected)"
                    )
                return "ok"

        session = FakeSession()
        with caplog.at_level(logging.WARNING):
            result = await call_tool(session, "create_character", {})

        assert result == "ok"
        assert session.calls == 2
        assert schema_drift_detected() is True
        # The warning must name the cause and the remedy, not just the error.
        assert "run-dmcp" in caplog.text
        assert "24" in caplog.text

    @pytest.mark.asyncio
    async def test_warns_only_once_across_many_calls(self, caplog):
        """A 568-room load must not emit one warning per character."""
        class FakeSession:
            """Drifts on every first attempt; the post-degrade retry succeeds."""

            def __init__(self):
                self.attempts = 0

            async def validate_tool_result(self, name, result):
                return None

            async def call_tool(self, name, arguments):
                self.attempts += 1
                if self.attempts % 2 == 1:
                    raise RuntimeError("Invalid structured content returned by tool x")
                return "ok"

        session = FakeSession()
        with caplog.at_level(logging.WARNING):
            for _ in range(3):
                await call_tool(session, "create_character", {})

        assert session.attempts == 6

        warnings = [r for r in caplog.records if "run-dmcp" in r.getMessage()]
        assert len(warnings) == 1

    @pytest.mark.asyncio
    async def test_characters_survive_a_stale_server(self):
        """End to end: a drifting server still yields characters."""
        game_data = {
            "title": "Stale",
            "rooms": [{
                "name": "R", "description": "r", "exits": [],
                "characters": ["Raskolnikov", "Sonia"],
            }],
        }

        class FakeSession:
            def __init__(self):
                self.made = []

            async def validate_tool_result(self, name, result):
                return None

            async def call_tool(self, name, arguments):
                if name == "create_character" and name not in self.made:
                    self.made.append(name)
                    raise RuntimeError("Invalid structured content returned by tool create_character")
                result = MagicMock()
                result.content = [MagicMock(text=json.dumps({"id": "id-1"}))]
                return result

        result = await load_game_with_session(FakeSession(), game_data)

        assert len(result["character_ids"]) == 2

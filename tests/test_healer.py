"""Unit tests for the MapHealer class."""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from heal_map import MapHealer, get_llm_config


class TestMapHealer:
    """Tests for MapHealer functionality."""

    @pytest.fixture
    def sample_rooms(self):
        """Sample rooms data for testing."""
        return {
            "title": "Test Map",
            "rooms": [
                {"name": "Room A", "exits": ["Room B", "Nonexistent"]},
                {"name": "Room B", "exits": []}
            ]
        }

    @pytest.fixture
    def healed_response(self):
        """Sample healed response from LLM."""
        return {
            "rooms": [
                {"name": "Room A", "exits": ["Room B", "Room C"]},
                {"name": "Room B", "exits": ["Room A"]},
                {"name": "Room C", "exits": ["Room A"]}
            ],
            "healing_summary": {
                "rooms_added": ["Room C"],
                "connections_added": ["Room A <-> Room B", "Room A <-> Room C"],
                "connections_removed": ["Room A -> Nonexistent"],
                "puzzle_gates": []
            }
        }

    def test_extract_json_direct(self):
        """Test extracting JSON from a direct JSON response."""
        healer = MapHealer("http://test", "key", "model")
        response = '{"rooms": [{"name": "Test"}]}'
        result = healer._extract_json(response)
        assert result["rooms"][0]["name"] == "Test"

    def test_extract_json_markdown_block(self):
        """Test extracting JSON from markdown code block."""
        healer = MapHealer("http://test", "key", "model")
        response = '''Here's the healed map:

```json
{"rooms": [{"name": "Test"}]}
```

Hope this helps!'''
        result = healer._extract_json(response)
        assert result["rooms"][0]["name"] == "Test"

    def test_extract_json_plain_code_block(self):
        """Test extracting JSON from plain code block."""
        healer = MapHealer("http://test", "key", "model")
        response = '''```
{"rooms": [{"name": "Test"}]}
```'''
        result = healer._extract_json(response)
        assert result["rooms"][0]["name"] == "Test"

    def test_extract_json_embedded(self):
        """Test extracting JSON embedded in text."""
        healer = MapHealer("http://test", "key", "model")
        response = '''I've healed the map. Here it is: {"rooms": [{"name": "Test"}]} That should work!'''
        result = healer._extract_json(response)
        assert result["rooms"][0]["name"] == "Test"

    def test_extract_json_nested(self):
        """Test extracting deeply nested JSON."""
        healer = MapHealer("http://test", "key", "model")
        response = '''{"rooms": [{"name": "Test", "data": {"nested": {"value": 1}}}]}'''
        result = healer._extract_json(response)
        assert result["rooms"][0]["data"]["nested"]["value"] == 1

    def test_extract_json_invalid(self):
        """Test that invalid JSON raises an error."""
        healer = MapHealer("http://test", "key", "model")
        response = "This is not JSON at all"
        with pytest.raises(ValueError, match="Could not extract valid JSON"):
            healer._extract_json(response)

    @patch('heal_map.MapHealer._call_llm')
    def test_heal_map_integration(self, mock_call_llm, sample_rooms, healed_response):
        """Test the full heal_map flow with mocked LLM."""
        mock_call_llm.return_value = json.dumps(healed_response)

        healer = MapHealer("http://test", "key", "model")
        result = healer.heal_map(sample_rooms, "Room A")

        assert len(result["rooms"]) == 3
        assert "Room C" in [r["name"] for r in result["rooms"]]
        assert result["healing_summary"]["rooms_added"] == ["Room C"]

    def test_healer_initialization(self):
        """Test MapHealer initializes correctly."""
        healer = MapHealer("http://localhost:8000/v1", "test-key", "test-model")
        assert healer.model == "test-model"


class TestGetLLMConfig:
    """Tests for LLM configuration."""

    @patch.dict('os.environ', {
        'OPENAI_API_KEY': 'sk-test-key',
        'OPENAI_MODEL': 'gpt-4o',
        'OPENAI_BASE_URL': 'https://api.openai.com/v1'
    })
    def test_remote_config(self):
        """Test remote (OpenAI) configuration."""
        api_base, api_key, model, mode = get_llm_config(use_remote=True)
        assert api_base == "https://api.openai.com/v1"
        assert api_key == "sk-test-key"
        assert model == "gpt-4o"
        assert "remote" in mode

    @patch.dict('os.environ', {
        'LLM_BASE_URL': 'http://localhost:1234/v1',
        'LLM_MODEL': 'local-model',
        'LLM_API_KEY': 'lm-studio'
    })
    def test_local_config(self):
        """Test local LLM configuration."""
        api_base, api_key, model, mode = get_llm_config(use_remote=False)
        assert api_base == "http://localhost:1234/v1"
        assert api_key == "lm-studio"
        assert model == "local-model"
        assert "local" in mode


class TestHealerWithRealAnalyzer:
    """Tests that verify healer integrates correctly with analyzer."""

    @pytest.fixture
    def disconnected_map(self):
        """Map with disconnected areas, as raw extractions typically produce."""
        return {
            "title": "Disconnected Test",
            "rooms": [
                {"name": "Start", "exits": ["Middle"]},
                {"name": "Middle", "exits": ["Start"]},
                {"name": "Isolated A", "exits": ["Isolated B"]},
                {"name": "Isolated B", "exits": ["Isolated A"]}
            ]
        }

    @patch('heal_map.MapHealer._call_llm')
    def test_detects_disconnected_areas(self, mock_call_llm, disconnected_map):
        """Test that healer correctly identifies disconnected areas."""
        # The LLM should receive info about disconnected areas
        healed_response = {
            "rooms": [
                {"name": "Start", "exits": ["Middle", "Bridge"]},
                {"name": "Middle", "exits": ["Start"]},
                {"name": "Bridge", "exits": ["Start", "Isolated A"]},
                {"name": "Isolated A", "exits": ["Isolated B", "Bridge"]},
                {"name": "Isolated B", "exits": ["Isolated A"]}
            ],
            "healing_summary": {
                "rooms_added": ["Bridge"],
                "connections_added": ["Start <-> Bridge", "Bridge <-> Isolated A"],
                "connections_removed": [],
                "puzzle_gates": []
            }
        }
        mock_call_llm.return_value = json.dumps(healed_response)

        healer = MapHealer("http://test", "key", "model")
        result = healer.heal_map(disconnected_map, "Start")

        # Verify the prompt included disconnected area info
        call_args = mock_call_llm.call_args[0][0]
        assert "Isolated A" in call_args
        assert "disconnected" in call_args.lower() or "subgraph" in call_args.lower()

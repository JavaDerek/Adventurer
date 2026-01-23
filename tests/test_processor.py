"""
Unit tests for the Doctor Who transcript processor.
Uses mocks to avoid actual LLM calls.
"""

import json
import pytest
from unittest.mock import Mock, MagicMock, patch, mock_open
from io import BytesIO

# Import the processor class
import sys
sys.path.insert(0, '.')
from process_transcript_full import FullContextProcessor


class TestDeduplicateRooms:
    """Tests for the deduplicate_rooms method"""

    def test_no_duplicates(self):
        """Rooms with unique names should all be kept"""
        processor = FullContextProcessor.__new__(FullContextProcessor)

        rooms = [
            {"name": "TARDIS", "description": "Time machine"},
            {"name": "Panopticon", "description": "Great hall"},
            {"name": "Matrix", "description": "Virtual reality"}
        ]

        result = processor.deduplicate_rooms(rooms)

        assert len(result) == 3
        assert {r["name"] for r in result} == {"TARDIS", "Panopticon", "Matrix"}

    def test_exact_duplicates_merged(self):
        """Rooms with identical names should be merged"""
        processor = FullContextProcessor.__new__(FullContextProcessor)

        rooms = [
            {"name": "TARDIS", "description": "Short desc", "items": ["console"]},
            {"name": "TARDIS", "description": "A longer description here", "items": ["time rotor"]}
        ]

        result = processor.deduplicate_rooms(rooms)

        assert len(result) == 1
        assert result[0]["name"] == "TARDIS"
        # Should keep longer description
        assert result[0]["description"] == "A longer description here"
        # Should merge items
        assert "console" in result[0]["items"]
        assert "time rotor" in result[0]["items"]

    def test_empty_name_skipped(self):
        """Rooms with empty names should be skipped"""
        processor = FullContextProcessor.__new__(FullContextProcessor)

        rooms = [
            {"name": "", "description": "No name"},
            {"name": "TARDIS", "description": "Valid room"},
            {"name": "   ", "description": "Whitespace name"}
        ]

        result = processor.deduplicate_rooms(rooms)

        assert len(result) == 1
        assert result[0]["name"] == "TARDIS"

    def test_merge_lists_deduplicates(self):
        """Merging should deduplicate list items"""
        processor = FullContextProcessor.__new__(FullContextProcessor)

        rooms = [
            {"name": "TARDIS", "characters": ["Doctor", "Sarah"]},
            {"name": "TARDIS", "characters": ["Doctor", "K-9"]}
        ]

        result = processor.deduplicate_rooms(rooms)

        assert len(result) == 1
        chars = result[0]["characters"]
        assert len(chars) == 3
        assert "Doctor" in chars
        assert "Sarah" in chars
        assert "K-9" in chars

    def test_empty_rooms_list(self):
        """Empty input should return empty output"""
        processor = FullContextProcessor.__new__(FullContextProcessor)

        result = processor.deduplicate_rooms([])

        assert result == []


class TestEnhanceWithMetadata:
    """Tests for the enhance_with_metadata method"""

    def test_basic_metadata(self):
        """Should add proper metadata structure"""
        processor = FullContextProcessor.__new__(FullContextProcessor)

        rooms = [
            {"name": "TARDIS", "characters": ["Doctor"]},
            {"name": "Panopticon", "characters": ["Doctor", "Chancellor"]}
        ]

        result = processor.enhance_with_metadata(rooms, "Test Episode")

        assert result["title"] == "Test Episode"
        assert result["format"] == "interactive_fiction_v1"
        assert result["source"] == "doctor_who_transcript"
        assert result["room_count"] == 2
        assert result["rooms"] == rooms

    def test_all_characters_extracted(self):
        """Should extract all unique characters"""
        processor = FullContextProcessor.__new__(FullContextProcessor)

        rooms = [
            {"name": "Room1", "characters": ["Doctor", "Sarah"]},
            {"name": "Room2", "characters": ["Doctor", "K-9"]},
            {"name": "Room3", "characters": []}
        ]

        result = processor.enhance_with_metadata(rooms, "Test")

        chars = result["metadata"]["all_characters"]
        assert len(chars) == 3
        assert "Doctor" in chars
        assert "Sarah" in chars
        assert "K-9" in chars

    def test_all_locations_listed(self):
        """Should list all location names"""
        processor = FullContextProcessor.__new__(FullContextProcessor)

        rooms = [
            {"name": "TARDIS"},
            {"name": "Panopticon"},
            {"name": "Matrix"}
        ]

        result = processor.enhance_with_metadata(rooms, "Test")

        locations = result["metadata"]["all_locations"]
        assert locations == ["TARDIS", "Panopticon", "Matrix"]

    def test_empty_rooms(self):
        """Should handle empty rooms list"""
        processor = FullContextProcessor.__new__(FullContextProcessor)

        result = processor.enhance_with_metadata([], "Empty Episode")

        assert result["room_count"] == 0
        assert result["rooms"] == []
        assert result["metadata"]["all_characters"] == []
        assert result["metadata"]["all_locations"] == []


class TestExtractTextFromPdf:
    """Tests for PDF text extraction"""

    @patch('process_transcript_full.PyPDF2.PdfReader')
    def test_extracts_all_pages(self, mock_reader_class):
        """Should extract text from all PDF pages"""
        # Setup mock
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Page 1 content"
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "Page 2 content"

        mock_reader = Mock()
        mock_reader.pages = [mock_page1, mock_page2]
        mock_reader_class.return_value = mock_reader

        processor = FullContextProcessor.__new__(FullContextProcessor)

        with patch('builtins.open', mock_open()):
            result = processor.extract_text_from_pdf("test.pdf")

        assert "Page 1 content" in result
        assert "Page 2 content" in result

    @patch('process_transcript_full.PyPDF2.PdfReader')
    def test_joins_pages_with_newlines(self, mock_reader_class):
        """Should join pages with double newlines"""
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "First"
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "Second"

        mock_reader = Mock()
        mock_reader.pages = [mock_page1, mock_page2]
        mock_reader_class.return_value = mock_reader

        processor = FullContextProcessor.__new__(FullContextProcessor)

        with patch('builtins.open', mock_open()):
            result = processor.extract_text_from_pdf("test.pdf")

        assert result == "First\n\nSecond"


class TestExtractAllRooms:
    """Tests for LLM-based room extraction with mocked API"""

    def test_parses_valid_json_response(self):
        """Should parse valid JSON array response"""
        processor = FullContextProcessor.__new__(FullContextProcessor)
        processor.model = "test-model"

        # Mock the streaming response
        mock_chunk = Mock()
        mock_chunk.choices = [Mock()]
        mock_chunk.choices[0].delta.content = json.dumps([
            {"name": "TARDIS", "description": "Time machine", "exits": [],
             "items": [], "characters": [], "events": [], "atmosphere": "mysterious"}
        ])

        mock_stream = [mock_chunk]

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = iter(mock_stream)
        processor.client = mock_client

        result = processor.extract_all_rooms("Sample transcript text")

        assert len(result) == 1
        assert result[0]["name"] == "TARDIS"

    def test_strips_markdown_code_blocks(self):
        """Should extract JSON from markdown code blocks"""
        processor = FullContextProcessor.__new__(FullContextProcessor)
        processor.model = "test-model"

        json_content = json.dumps([{"name": "TARDIS", "description": "Test",
                                    "exits": [], "items": [], "characters": [],
                                    "events": [], "atmosphere": "mysterious"}])

        # Simulate response with markdown wrapper
        mock_chunk = Mock()
        mock_chunk.choices = [Mock()]
        mock_chunk.choices[0].delta.content = f"```json\n{json_content}\n```"

        mock_stream = [mock_chunk]

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = iter(mock_stream)
        processor.client = mock_client

        result = processor.extract_all_rooms("Sample text")

        assert len(result) == 1
        assert result[0]["name"] == "TARDIS"

    def test_handles_streaming_chunks(self):
        """Should accumulate multiple streaming chunks"""
        processor = FullContextProcessor.__new__(FullContextProcessor)
        processor.model = "test-model"

        # Split JSON across multiple chunks
        json_str = json.dumps([{"name": "TARDIS", "description": "Test",
                               "exits": [], "items": [], "characters": [],
                               "events": [], "atmosphere": "mysterious"}])

        chunks = []
        for char in json_str:
            mock_chunk = Mock()
            mock_chunk.choices = [Mock()]
            mock_chunk.choices[0].delta.content = char
            chunks.append(mock_chunk)

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = iter(chunks)
        processor.client = mock_client

        result = processor.extract_all_rooms("Sample text")

        assert len(result) == 1
        assert result[0]["name"] == "TARDIS"

    def test_handles_empty_delta_content(self):
        """Should handle chunks with no content"""
        processor = FullContextProcessor.__new__(FullContextProcessor)
        processor.model = "test-model"

        json_content = json.dumps([{"name": "TARDIS", "description": "Test",
                                    "exits": [], "items": [], "characters": [],
                                    "events": [], "atmosphere": "mysterious"}])

        # Mix of empty and content chunks
        mock_chunk1 = Mock()
        mock_chunk1.choices = [Mock()]
        mock_chunk1.choices[0].delta.content = None

        mock_chunk2 = Mock()
        mock_chunk2.choices = [Mock()]
        mock_chunk2.choices[0].delta.content = json_content

        mock_chunk3 = Mock()
        mock_chunk3.choices = [Mock()]
        mock_chunk3.choices[0].delta.content = None

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = iter([mock_chunk1, mock_chunk2, mock_chunk3])
        processor.client = mock_client

        result = processor.extract_all_rooms("Sample text")

        assert len(result) == 1

    def test_deduplicates_results(self):
        """Should deduplicate rooms after extraction"""
        processor = FullContextProcessor.__new__(FullContextProcessor)
        processor.model = "test-model"

        # Response with duplicate room names
        json_content = json.dumps([
            {"name": "TARDIS", "description": "Short", "exits": [],
             "items": ["console"], "characters": [], "events": [], "atmosphere": "mysterious"},
            {"name": "TARDIS", "description": "Longer description", "exits": [],
             "items": ["time rotor"], "characters": [], "events": [], "atmosphere": "mysterious"}
        ])

        mock_chunk = Mock()
        mock_chunk.choices = [Mock()]
        mock_chunk.choices[0].delta.content = json_content

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = iter([mock_chunk])
        processor.client = mock_client

        result = processor.extract_all_rooms("Sample text")

        # Should be deduplicated to 1 room
        assert len(result) == 1
        assert result[0]["name"] == "TARDIS"
        # Should have merged items
        assert "console" in result[0]["items"]
        assert "time rotor" in result[0]["items"]


class TestProcessPdf:
    """Integration tests for the full processing pipeline"""

    @patch.object(FullContextProcessor, 'extract_text_from_pdf')
    @patch.object(FullContextProcessor, 'extract_all_rooms')
    def test_full_pipeline(self, mock_extract_rooms, mock_extract_text):
        """Should run full pipeline and save output"""
        mock_extract_text.return_value = "Sample transcript"
        mock_extract_rooms.return_value = [
            {"name": "TARDIS", "description": "Test", "exits": [],
             "items": [], "characters": ["Doctor"], "events": [], "atmosphere": "mysterious"}
        ]

        processor = FullContextProcessor.__new__(FullContextProcessor)

        with patch('builtins.open', mock_open()) as mock_file:
            result = processor.process_pdf("test.pdf", "output.json")

        assert result is not None
        assert result["room_count"] == 1
        assert result["rooms"][0]["name"] == "TARDIS"

        # Verify file was written
        mock_file.assert_called_with("output.json", 'w', encoding='utf-8')

    @patch.object(FullContextProcessor, 'extract_text_from_pdf')
    @patch.object(FullContextProcessor, 'extract_all_rooms')
    def test_returns_none_on_no_rooms(self, mock_extract_rooms, mock_extract_text):
        """Should return None if no rooms extracted"""
        mock_extract_text.return_value = "Sample transcript"
        mock_extract_rooms.return_value = []

        processor = FullContextProcessor.__new__(FullContextProcessor)

        result = processor.process_pdf("test.pdf", "output.json")

        assert result is None

    @patch.object(FullContextProcessor, 'extract_text_from_pdf')
    @patch.object(FullContextProcessor, 'extract_all_rooms')
    def test_default_output_path(self, mock_extract_rooms, mock_extract_text):
        """Should generate default output path from input filename"""
        mock_extract_text.return_value = "Sample transcript"
        mock_extract_rooms.return_value = [
            {"name": "TARDIS", "description": "Test", "exits": [],
             "items": [], "characters": [], "events": [], "atmosphere": "mysterious"}
        ]

        processor = FullContextProcessor.__new__(FullContextProcessor)

        with patch('builtins.open', mock_open()) as mock_file:
            result = processor.process_pdf("My Episode.pdf")

        # Should use stem + _rooms_full.json
        mock_file.assert_called_with("My Episode_rooms_full.json", 'w', encoding='utf-8')


class TestJsonExtraction:
    """Tests for JSON extraction from various response formats"""

    def test_extracts_from_triple_backtick_json(self):
        """Should extract JSON from ```json ... ``` blocks"""
        processor = FullContextProcessor.__new__(FullContextProcessor)
        processor.model = "test-model"

        response = '```json\n[{"name": "Test"}]\n```'

        mock_chunk = Mock()
        mock_chunk.choices = [Mock()]
        mock_chunk.choices[0].delta.content = response

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = iter([mock_chunk])
        processor.client = mock_client

        result = processor.extract_all_rooms("text")

        assert len(result) == 1
        assert result[0]["name"] == "Test"

    def test_extracts_from_plain_backticks(self):
        """Should extract JSON from plain ``` ... ``` blocks"""
        processor = FullContextProcessor.__new__(FullContextProcessor)
        processor.model = "test-model"

        response = '```\n[{"name": "Test"}]\n```'

        mock_chunk = Mock()
        mock_chunk.choices = [Mock()]
        mock_chunk.choices[0].delta.content = response

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = iter([mock_chunk])
        processor.client = mock_client

        result = processor.extract_all_rooms("text")

        assert len(result) == 1

    def test_handles_raw_json_array(self):
        """Should handle raw JSON without markdown"""
        processor = FullContextProcessor.__new__(FullContextProcessor)
        processor.model = "test-model"

        response = '[{"name": "Test"}]'

        mock_chunk = Mock()
        mock_chunk.choices = [Mock()]
        mock_chunk.choices[0].delta.content = response

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = iter([mock_chunk])
        processor.client = mock_client

        result = processor.extract_all_rooms("text")

        assert len(result) == 1

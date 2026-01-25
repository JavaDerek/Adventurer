"""
Unit tests for the fiction-to-interactive-fiction processor.
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
        assert result["source"] == "fiction"
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


class TestExtractJsonFromResponse:
    """Tests for the extract_json_from_response fallback method"""

    def test_direct_json_array(self):
        """Should parse clean JSON array directly"""
        processor = FullContextProcessor.__new__(FullContextProcessor)
        response = '[{"name": "TARDIS"}, {"name": "Panopticon"}]'

        result = processor.extract_json_from_response(response)

        assert len(result) == 2
        assert result[0]["name"] == "TARDIS"

    def test_json_object_with_rooms_key(self):
        """Should extract rooms array from object with rooms key"""
        processor = FullContextProcessor.__new__(FullContextProcessor)
        response = '{"rooms": [{"name": "TARDIS"}]}'

        result = processor.extract_json_from_response(response)

        assert len(result) == 1
        assert result[0]["name"] == "TARDIS"

    def test_markdown_json_block(self):
        """Should extract from ```json code blocks"""
        processor = FullContextProcessor.__new__(FullContextProcessor)
        response = 'Here is the JSON:\n```json\n[{"name": "TARDIS"}]\n```\nDone!'

        result = processor.extract_json_from_response(response)

        assert len(result) == 1
        assert result[0]["name"] == "TARDIS"

    def test_plain_markdown_block(self):
        """Should extract from plain ``` code blocks"""
        processor = FullContextProcessor.__new__(FullContextProcessor)
        response = '```\n[{"name": "TARDIS"}]\n```'

        result = processor.extract_json_from_response(response)

        assert len(result) == 1
        assert result[0]["name"] == "TARDIS"

    def test_json_array_with_surrounding_text(self):
        """Should find JSON array boundaries in mixed text"""
        processor = FullContextProcessor.__new__(FullContextProcessor)
        response = 'Here are the rooms: [{"name": "TARDIS"}, {"name": "Matrix"}] Hope this helps!'

        result = processor.extract_json_from_response(response)

        assert len(result) == 2

    def test_rooms_object_with_surrounding_text(self):
        """Should find {"rooms": ...} object in mixed text"""
        processor = FullContextProcessor.__new__(FullContextProcessor)
        response = 'Output: {"rooms": [{"name": "TARDIS"}]} end'

        result = processor.extract_json_from_response(response)

        assert len(result) == 1

    def test_invalid_json_returns_empty(self):
        """Should return empty list for unparseable content"""
        processor = FullContextProcessor.__new__(FullContextProcessor)
        response = 'This is not JSON at all'

        result = processor.extract_json_from_response(response)

        assert result == []

    def test_nested_brackets_handled(self):
        """Should handle nested JSON structures correctly"""
        processor = FullContextProcessor.__new__(FullContextProcessor)
        response = '[{"name": "TARDIS", "items": ["console", "lever"]}]'

        result = processor.extract_json_from_response(response)

        assert len(result) == 1
        assert result[0]["items"] == ["console", "lever"]


class TestGetLlmConfig:
    """Tests for the get_llm_config function"""

    @patch.dict('os.environ', {
        'OPENAI_API_KEY': 'sk-test-key',
        'OPENAI_MODEL': 'gpt-4o',
        'OPENAI_BASE_URL': 'https://api.openai.com/v1'
    })
    def test_remote_config(self):
        """Should return OpenAI config when use_remote=True"""
        from process_transcript_full import get_llm_config

        api_base, api_key, model, mode_name = get_llm_config(use_remote=True)

        assert api_base == 'https://api.openai.com/v1'
        assert api_key == 'sk-test-key'
        assert model == 'gpt-4o'
        assert 'remote' in mode_name.lower()

    @patch.dict('os.environ', {
        'LLM_BASE_URL': 'http://localhost:1234/v1',
        'LLM_API_KEY': 'lm-studio',
        'LLM_MODEL': 'qwen2.5-32b-instruct'
    })
    def test_local_config(self):
        """Should return local LLM config when use_remote=False"""
        from process_transcript_full import get_llm_config

        api_base, api_key, model, mode_name = get_llm_config(use_remote=False)

        assert api_base == 'http://localhost:1234/v1'
        assert api_key == 'lm-studio'
        assert model == 'qwen2.5-32b-instruct'
        assert 'local' in mode_name.lower()

    @patch.dict('os.environ', {'OPENAI_API_KEY': '', 'OPENAI_MODEL': 'gpt-4o'}, clear=True)
    def test_remote_missing_key_exits(self):
        """Should exit if OPENAI_API_KEY is missing for remote"""
        from process_transcript_full import get_llm_config

        with pytest.raises(SystemExit):
            get_llm_config(use_remote=True)

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-your-key-here'}, clear=True)
    def test_remote_placeholder_key_exits(self):
        """Should exit if OPENAI_API_KEY is still placeholder"""
        from process_transcript_full import get_llm_config

        with pytest.raises(SystemExit):
            get_llm_config(use_remote=True)


class TestExtractAllRoomsErrorHandling:
    """Tests for error handling in extract_all_rooms"""

    def test_json_decode_error_triggers_fallback(self):
        """Should use fallback parser when initial JSON parse fails"""
        processor = FullContextProcessor.__new__(FullContextProcessor)
        processor.model = "test-model"

        # Response with text before JSON (will fail initial parse but fallback succeeds)
        response = 'Here is the data: [{"name": "TARDIS"}]'

        mock_chunk = Mock()
        mock_chunk.choices = [Mock()]
        mock_chunk.choices[0].delta.content = response

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = iter([mock_chunk])
        processor.client = mock_client

        result = processor.extract_all_rooms("text")

        assert len(result) == 1
        assert result[0]["name"] == "TARDIS"

    def test_api_exception_returns_empty(self):
        """Should return empty list on API exception"""
        processor = FullContextProcessor.__new__(FullContextProcessor)
        processor.model = "test-model"

        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        processor.client = mock_client

        result = processor.extract_all_rooms("text")

        assert result == []

    def test_completely_invalid_response_returns_empty(self):
        """Should return empty list when all parsing strategies fail"""
        processor = FullContextProcessor.__new__(FullContextProcessor)
        processor.model = "test-model"

        # Response that can't be parsed as JSON at all
        response = 'I cannot extract any rooms from this transcript.'

        mock_chunk = Mock()
        mock_chunk.choices = [Mock()]
        mock_chunk.choices[0].delta.content = response

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = iter([mock_chunk])
        processor.client = mock_client

        result = processor.extract_all_rooms("text")

        assert result == []


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


class TestEstimateTokens:
    """Tests for the estimate_tokens method"""

    def test_basic_estimate(self):
        """Should estimate ~4 chars per token"""
        processor = FullContextProcessor.__new__(FullContextProcessor)

        # 400 characters should be ~100 tokens
        text = "a" * 400
        result = processor.estimate_tokens(text)

        assert result == 100

    def test_empty_text(self):
        """Should return 0 for empty text"""
        processor = FullContextProcessor.__new__(FullContextProcessor)

        result = processor.estimate_tokens("")

        assert result == 0

    def test_short_text(self):
        """Should handle text shorter than 4 chars"""
        processor = FullContextProcessor.__new__(FullContextProcessor)

        result = processor.estimate_tokens("abc")

        assert result == 0  # 3 // 4 = 0


class TestNeedsChunking:
    """Tests for the needs_chunking method"""

    @patch('process_transcript_full.CONTEXT_TOKEN_LIMIT', 1000)
    def test_needs_chunking_true(self):
        """Should return True for text exceeding limit"""
        processor = FullContextProcessor.__new__(FullContextProcessor)

        # 5000 chars = 1250 tokens > 1000 limit
        text = "a" * 5000
        result = processor.needs_chunking(text)

        assert result is True

    @patch('process_transcript_full.CONTEXT_TOKEN_LIMIT', 1000)
    def test_needs_chunking_false(self):
        """Should return False for text under limit"""
        processor = FullContextProcessor.__new__(FullContextProcessor)

        # 2000 chars = 500 tokens < 1000 limit
        text = "a" * 2000
        result = processor.needs_chunking(text)

        assert result is False


class TestDetectChapterMarkers:
    """Tests for the detect_chapter_markers method"""

    def test_detect_part_markers(self):
        """Should detect PART markers"""
        processor = FullContextProcessor.__new__(FullContextProcessor)

        text = "Introduction\n\nPART ONE\n\nChapter content\n\nPART TWO\n\nMore content"
        markers = processor.detect_chapter_markers(text)

        assert len(markers) >= 2
        marker_texts = [m[1] for m in markers]
        assert any("PART" in m for m in marker_texts)

    def test_detect_chapter_markers(self):
        """Should detect CHAPTER markers"""
        processor = FullContextProcessor.__new__(FullContextProcessor)

        text = "Book start\n\nCHAPTER 1\n\nContent\n\nCHAPTER 2\n\nMore"
        markers = processor.detect_chapter_markers(text)

        assert len(markers) >= 2

    def test_detect_mixed_case(self):
        """Should detect mixed case markers"""
        processor = FullContextProcessor.__new__(FullContextProcessor)

        text = "Start\n\nPart One\n\nContent\n\nChapter 1\n\nMore"
        markers = processor.detect_chapter_markers(text)

        assert len(markers) >= 2

    def test_detect_roman_numerals(self):
        """Should detect Roman numeral chapters"""
        processor = FullContextProcessor.__new__(FullContextProcessor)

        text = "Start\n\nPART I\n\nContent\n\nCHAPTER II\n\nMore"
        markers = processor.detect_chapter_markers(text)

        assert len(markers) >= 2

    def test_detect_epilogue_prologue(self):
        """Should detect EPILOGUE and PROLOGUE"""
        processor = FullContextProcessor.__new__(FullContextProcessor)

        text = "PROLOGUE\n\nBeginning\n\nContent\n\nEPILOGUE\n\nEnd"
        markers = processor.detect_chapter_markers(text)

        assert len(markers) >= 2
        marker_texts = [m[1].upper() for m in markers]
        assert "PROLOGUE" in marker_texts
        assert "EPILOGUE" in marker_texts

    def test_detect_with_page_numbers(self):
        """Should detect markers with PDF page number prefixes"""
        processor = FullContextProcessor.__new__(FullContextProcessor)

        text = "6 of 967 PART I\n\nContent\n\n50 of 967 PART II\n\nMore"
        markers = processor.detect_chapter_markers(text)

        assert len(markers) >= 2

    def test_no_markers_returns_empty(self):
        """Should return empty list when no markers found"""
        processor = FullContextProcessor.__new__(FullContextProcessor)

        text = "Just some regular text without any chapter markers."
        markers = processor.detect_chapter_markers(text)

        assert markers == []

    def test_markers_sorted_by_position(self):
        """Should return markers sorted by position"""
        processor = FullContextProcessor.__new__(FullContextProcessor)

        text = "PART ONE\n\nContent\n\nPART TWO\n\nMore\n\nPART THREE\n\nEnd"
        markers = processor.detect_chapter_markers(text)

        positions = [m[0] for m in markers]
        assert positions == sorted(positions)


class TestSplitAtMarkers:
    """Tests for the split_at_markers method"""

    @patch('process_transcript_full.MIN_CHUNK_TOKENS', 1)
    def test_split_at_markers_basic(self):
        """Should split text at marker positions"""
        processor = FullContextProcessor.__new__(FullContextProcessor)

        text = "Intro content\n\nPART ONE\n\nChapter content here"
        markers = [(15, "PART ONE")]

        chunks = processor.split_at_markers(text, markers)

        assert len(chunks) == 2
        assert "Intro" in chunks[0]['text']
        assert "PART ONE" in chunks[1]['text']

    def test_split_empty_markers(self):
        """Should return single chunk when no markers"""
        processor = FullContextProcessor.__new__(FullContextProcessor)

        text = "All the content in one piece"
        markers = []

        chunks = processor.split_at_markers(text, markers)

        assert len(chunks) == 1
        assert chunks[0]['text'] == text
        assert chunks[0]['start_marker'] is None

    @patch('process_transcript_full.MIN_CHUNK_TOKENS', 1)
    def test_chunk_indices_sequential(self):
        """Should have sequential chunk indices"""
        processor = FullContextProcessor.__new__(FullContextProcessor)

        text = "A\n\nPART ONE\n\nB\n\nPART TWO\n\nC"
        markers = [(3, "PART ONE"), (17, "PART TWO")]

        chunks = processor.split_at_markers(text, markers)

        indices = [c['chunk_index'] for c in chunks]
        assert indices == list(range(len(chunks)))


class TestSplitByTokens:
    """Tests for the split_by_tokens method"""

    @patch('process_transcript_full.CONTEXT_TOKEN_LIMIT', 100)
    @patch('process_transcript_full.CHUNK_OVERLAP_TOKENS', 10)
    def test_split_by_tokens_basic(self):
        """Should split text into chunks by token count"""
        processor = FullContextProcessor.__new__(FullContextProcessor)

        # Create text that needs multiple chunks (100 token limit, ~400 chars)
        text = "This is a paragraph of content. " * 50  # ~1600 chars

        chunks = processor.split_by_tokens(text, target_tokens=50)

        assert len(chunks) > 1
        for chunk in chunks:
            assert 'text' in chunk
            assert 'estimated_tokens' in chunk
            assert 'chunk_index' in chunk

    def test_split_by_tokens_short_text(self):
        """Should return single chunk for short text"""
        processor = FullContextProcessor.__new__(FullContextProcessor)

        text = "Short text"

        chunks = processor.split_by_tokens(text, target_tokens=1000)

        assert len(chunks) == 1
        assert chunks[0]['text'] == text

    @patch('process_transcript_full.CHUNK_OVERLAP_TOKENS', 10)
    def test_split_by_tokens_has_markers(self):
        """Should add token chunk markers"""
        processor = FullContextProcessor.__new__(FullContextProcessor)

        text = "word " * 500  # Long enough to split

        chunks = processor.split_by_tokens(text, target_tokens=100)

        for chunk in chunks:
            assert 'Token chunk' in chunk['start_marker']


class TestSmartSplit:
    """Tests for the smart_split method"""

    @patch('process_transcript_full.MIN_CHUNK_TOKENS', 1)
    def test_smart_split_uses_markers_when_found(self):
        """Should use marker-based splitting when markers exist"""
        processor = FullContextProcessor.__new__(FullContextProcessor)

        text = "Intro\n\nPART ONE\n\nContent\n\nPART TWO\n\nMore content"

        with patch('builtins.print'):  # Suppress print output
            chunks = processor.smart_split(text)

        # Should have split at PART markers
        assert len(chunks) >= 2

    def test_smart_split_falls_back_to_tokens(self):
        """Should use token splitting when no markers found"""
        processor = FullContextProcessor.__new__(FullContextProcessor)

        # Text without any chapter markers
        text = "Regular content without markers. " * 100

        with patch('builtins.print'):
            chunks = processor.smart_split(text)

        # Should have created chunks
        assert len(chunks) >= 1
        # Token chunks have specific marker format
        assert all('chunk' in str(c.get('start_marker', '')).lower() or c.get('start_marker') is None
                   for c in chunks)


class TestExtractRoomsFromChunk:
    """Tests for the extract_rooms_from_chunk method"""

    def test_extract_rooms_from_chunk_basic(self):
        """Should extract rooms from a chunk"""
        processor = FullContextProcessor.__new__(FullContextProcessor)
        processor.model = "test-model"

        mock_chunk = Mock()
        mock_chunk.choices = [Mock()]
        mock_chunk.choices[0].delta.content = json.dumps([
            {"name": "Room A", "description": "First room", "exits": [],
             "items": [], "characters": [], "events": [], "atmosphere": "neutral"}
        ])

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = iter([mock_chunk])
        processor.client = mock_client

        chunk = {'text': 'Some content', 'start_marker': 'PART ONE', 'chunk_index': 0, 'estimated_tokens': 100}

        with patch('builtins.print'):
            result = processor.extract_rooms_from_chunk(chunk)

        assert len(result) == 1
        assert result[0]["name"] == "Room A"

    def test_extract_rooms_from_chunk_with_context(self):
        """Should include context about previous rooms"""
        processor = FullContextProcessor.__new__(FullContextProcessor)
        processor.model = "test-model"

        mock_chunk = Mock()
        mock_chunk.choices = [Mock()]
        mock_chunk.choices[0].delta.content = json.dumps([
            {"name": "Room B", "description": "Second room", "exits": [],
             "items": [], "characters": [], "events": [], "atmosphere": "neutral"}
        ])

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = iter([mock_chunk])
        processor.client = mock_client

        chunk = {'text': 'More content', 'start_marker': 'PART TWO', 'chunk_index': 1, 'estimated_tokens': 100}

        with patch('builtins.print'):
            result = processor.extract_rooms_from_chunk(chunk, chunk_context="Previously seen: Room A")

        assert len(result) == 1
        # Verify the API was called (context would be in the prompt)
        mock_client.chat.completions.create.assert_called_once()


class TestProcessChunked:
    """Tests for the process_chunked method"""

    @patch.object(FullContextProcessor, 'smart_split')
    @patch.object(FullContextProcessor, 'extract_rooms_from_chunk')
    @patch.object(FullContextProcessor, 'deduplicate_rooms')
    def test_process_chunked_basic(self, mock_dedup, mock_extract, mock_split):
        """Should process all chunks and merge results"""
        processor = FullContextProcessor.__new__(FullContextProcessor)

        mock_split.return_value = [
            {'text': 'Chunk 1', 'start_marker': 'PART ONE', 'chunk_index': 0, 'estimated_tokens': 100},
            {'text': 'Chunk 2', 'start_marker': 'PART TWO', 'chunk_index': 1, 'estimated_tokens': 100}
        ]

        mock_extract.side_effect = [
            [{"name": "Room A", "description": "First"}],
            [{"name": "Room B", "description": "Second"}]
        ]

        mock_dedup.return_value = [
            {"name": "Room A", "description": "First"},
            {"name": "Room B", "description": "Second"}
        ]

        with patch('builtins.print'):
            result = processor.process_chunked("Full text")

        assert mock_split.called
        assert mock_extract.call_count == 2
        assert mock_dedup.called
        assert len(result) == 2

    @patch.object(FullContextProcessor, 'smart_split')
    @patch.object(FullContextProcessor, 'extract_rooms_from_chunk')
    @patch.object(FullContextProcessor, 'deduplicate_rooms')
    def test_process_chunked_handles_duplicates(self, mock_dedup, mock_extract, mock_split):
        """Should deduplicate rooms from multiple chunks"""
        processor = FullContextProcessor.__new__(FullContextProcessor)

        mock_split.return_value = [
            {'text': 'Chunk 1', 'start_marker': 'PART ONE', 'chunk_index': 0, 'estimated_tokens': 100},
            {'text': 'Chunk 2', 'start_marker': 'PART TWO', 'chunk_index': 1, 'estimated_tokens': 100}
        ]

        # Both chunks return same room
        mock_extract.side_effect = [
            [{"name": "Room A", "description": "Short"}],
            [{"name": "Room A", "description": "Longer description"}]
        ]

        mock_dedup.return_value = [
            {"name": "Room A", "description": "Longer description"}
        ]

        with patch('builtins.print'):
            result = processor.process_chunked("Full text")

        # Should call deduplicate with all rooms
        mock_dedup.assert_called_once()
        assert len(result) == 1


class TestProcessPdfChunkedPath:
    """Tests for the chunked processing path in process_pdf"""

    @patch.object(FullContextProcessor, 'extract_text_from_pdf')
    @patch.object(FullContextProcessor, 'needs_chunking')
    @patch.object(FullContextProcessor, 'process_chunked')
    def test_process_pdf_uses_chunked_when_needed(self, mock_chunked, mock_needs, mock_extract):
        """Should use chunked processing for large documents"""
        mock_extract.return_value = "Large document text"
        mock_needs.return_value = True
        mock_chunked.return_value = [
            {"name": "Room A", "description": "Test", "exits": [],
             "items": [], "characters": [], "events": [], "atmosphere": "mysterious"}
        ]

        processor = FullContextProcessor.__new__(FullContextProcessor)

        with patch('builtins.open', mock_open()):
            with patch('builtins.print'):
                result = processor.process_pdf("test.pdf", "output.json")

        assert mock_needs.called
        assert mock_chunked.called
        assert result is not None

    @patch.object(FullContextProcessor, 'extract_text_from_pdf')
    @patch.object(FullContextProcessor, 'needs_chunking')
    @patch.object(FullContextProcessor, 'extract_all_rooms')
    def test_process_pdf_uses_single_pass_when_small(self, mock_extract_rooms, mock_needs, mock_extract):
        """Should use single-pass for small documents"""
        mock_extract.return_value = "Small document text"
        mock_needs.return_value = False
        mock_extract_rooms.return_value = [
            {"name": "Room A", "description": "Test", "exits": [],
             "items": [], "characters": [], "events": [], "atmosphere": "mysterious"}
        ]

        processor = FullContextProcessor.__new__(FullContextProcessor)

        with patch('builtins.open', mock_open()):
            result = processor.process_pdf("test.pdf", "output.json")

        assert mock_needs.called
        assert mock_extract_rooms.called
        assert result is not None


class TestCleanPdfText:
    """Tests for the clean_pdf_text method"""

    def test_clean_pdf_text_detects_encoding_issue(self):
        """Should detect and fix when 1s are used as spaces"""
        processor = FullContextProcessor.__new__(FullContextProcessor)

        # Text with 1s instead of spaces (common PDF encoding issue)
        # Need enough letter1letter patterns to trigger detection (>30%)
        text = "BOOK1ONE\n\nThis1is1some1text1with1encoding1issues1that1has1many1words"

        with patch('builtins.print'):
            result = processor.clean_pdf_text(text)

        assert "BOOK ONE" in result
        assert "This is some text with encoding issues" in result

    def test_clean_pdf_text_preserves_normal_text(self):
        """Should not modify text that doesn't have encoding issues"""
        processor = FullContextProcessor.__new__(FullContextProcessor)

        text = "BOOK ONE\n\nThis is normal text with the number 123 in it."
        result = processor.clean_pdf_text(text)

        # Should be unchanged (or minimally changed)
        assert "BOOK ONE" in result
        assert "normal text" in result

    def test_clean_pdf_text_handles_leading_1s(self):
        """Should convert leading 1s to indentation spaces when encoding issue detected"""
        processor = FullContextProcessor.__new__(FullContextProcessor)

        # Leading 1s that should be spaces, plus enough word1word patterns to trigger detection
        text = "11111BOOK1ONE\n111Content1here1with1more1encoding1issues1that1trigger1detection"

        with patch('builtins.print'):
            result = processor.clean_pdf_text(text)

        assert "BOOK ONE" in result
        assert "Content here" in result

    def test_clean_pdf_text_handles_punctuation(self):
        """Should fix 1s after punctuation and before letters when encoding issue detected"""
        processor = FullContextProcessor.__new__(FullContextProcessor)

        # Need enough letter1letter patterns to trigger detection
        text = "Hello.1World!1Test,1more1text1and1even1more1words1here1now"

        with patch('builtins.print'):
            result = processor.clean_pdf_text(text)

        assert "Hello. World" in result
        assert "more text" in result


class TestDetectChapterMarkersWithTocFiltering:
    """Tests for TOC filtering in detect_chapter_markers"""

    def test_filters_toc_entries_with_very_long_prose(self):
        """Should prefer chapter markers followed by prose over TOC entries"""
        processor = FullContextProcessor.__new__(FullContextProcessor)

        # Simulate a text with TOC entry and actual chapter with very long prose lines
        # The TOC section needs to be long enough that the prose lines are outside
        # the 1500 char lookahead window from the TOC marker
        short_lines = "\n".join(["  Short chapter name " + str(i) for i in range(100)])
        toc_section = f"""Table of Contents

BOOK ONE
{short_lines}

"""
        # Create prose lines that are definitely > 60 chars
        long_line = "A" * 80  # 80 chars, definitely > 60
        prose_section = f"""
BOOK ONE

{long_line}
{long_line}
{long_line}
{long_line}
{long_line}
{long_line}
{long_line}
{long_line}
{long_line}
{long_line}
"""
        text = toc_section + prose_section

        markers = processor.detect_chapter_markers(text)

        # Should have found at least one marker
        assert len(markers) >= 1
        # The marker should be the one from prose section (after TOC)
        for pos, marker in markers:
            assert pos > len(toc_section) - 50  # Allow some tolerance

    def test_handles_trailing_artifacts(self):
        """Should detect markers with trailing encoding artifacts"""
        processor = FullContextProcessor.__new__(FullContextProcessor)

        # Create long lines for prose detection (>60 chars each)
        long_line = "This is a long paragraph of prose content that demonstrates this is actual."

        # Text with trailing 1s (like from PDF encoding issues)
        text = f"""Start of book

BOOK ONE111

{long_line}
{long_line}
{long_line}
{long_line}
{long_line}
{long_line}
{long_line}
{long_line}
{long_line}
{long_line}
"""
        markers = processor.detect_chapter_markers(text)

        assert len(markers) >= 1
        assert any("BOOK ONE" in m[1] for m in markers)

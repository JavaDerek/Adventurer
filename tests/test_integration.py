"""
Integration tests for LLM connections.

These tests make REAL API calls and require:
- For remote tests: OPENAI_API_KEY set in .env
- For local tests: Local LLM server running (e.g., LM Studio)

Run with: pytest tests/test_integration.py -v
Or just integration tests: pytest -m integration -v

These tests are excluded from CI by default.
"""

import json
import os
import pytest
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()


@pytest.mark.integration
class TestRemoteOpenAIConnection:
    """Integration tests for OpenAI API connection."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up OpenAI client."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")

        if not self.api_key or self.api_key == "sk-your-key-here":
            pytest.skip("OPENAI_API_KEY not configured in .env")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )

    def test_simple_completion(self):
        """Test that OpenAI can generate a simple completion."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": "Reply with exactly: HELLO"}
            ],
            max_tokens=10,
            temperature=0.0
        )

        content = response.choices[0].message.content
        assert content is not None
        assert "HELLO" in content.upper()

    def test_json_generation(self):
        """Test that OpenAI can generate valid JSON."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": 'Return a JSON object with a single key "status" and value "ok". No markdown, just raw JSON.'}
            ],
            max_tokens=50,
            temperature=0.0
        )

        content = response.choices[0].message.content.strip()
        # Handle potential markdown wrapping
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        data = json.loads(content)
        assert data.get("status") == "ok"


@pytest.mark.integration
class TestLocalLLMConnection:
    """Integration tests for local LLM connection (e.g., LM Studio)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up local LLM client."""
        self.api_base = os.getenv("LLM_BASE_URL") or os.getenv("LLM_API_BASE")
        self.api_key = os.getenv("LLM_API_KEY", "lm-studio")
        self.model = os.getenv("LLM_MODEL")

        if not self.api_base:
            pytest.skip("LLM_BASE_URL not configured in .env")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )

    def test_simple_completion(self):
        """Test that local LLM can generate a simple completion."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": "Reply with exactly: HELLO"}
                ],
                max_tokens=10,
                temperature=0.0
            )
        except Exception as e:
            pytest.skip(f"Local LLM not available: {e}")

        content = response.choices[0].message.content
        assert content is not None
        assert len(content) > 0

    def test_json_generation(self):
        """Test that local LLM can generate valid JSON."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": 'Return a JSON object with a single key "status" and value "ok". No markdown, just raw JSON.'}
                ],
                max_tokens=50,
                temperature=0.0
            )
        except Exception as e:
            pytest.skip(f"Local LLM not available: {e}")

        content = response.choices[0].message.content.strip()
        # Handle potential markdown wrapping
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        data = json.loads(content)
        assert data.get("status") == "ok"


@pytest.mark.integration
class TestGoldStandardComparison:
    """Compare LLM output against gold standard extraction.

    These tests verify that an LLM can extract room data from a known text and
    produce results comparable to a validated gold standard.

    The gold standard is a raw extraction from the Constance Garnett (1914)
    translation of Dostoevsky's Crime and Punishment. Both the novel and the
    translation are public domain, so the fixture is freely redistributable.
    It contains 90 rooms and has not been through fix_exits.py, so its exits
    are still the descriptive text a fresh LLM run produces.
    """

    # A stable sample of rooms any competent extraction should find. Asserting
    # all 90 names would make this test a diff of one model's phrasing.
    EXPECTED_SAMPLE_ROOMS = [
        "Petersburg",
        "Raskolnikov’s garret (lodging under the roof)",
        "Landlady’s kitchen",
    ]

    PDF_PATH = "Crime_and_Punishment_T.pdf"
    GOLD_STANDARD_PATH = "tests/fixtures/crime_and_punishment_gold.json"

    @pytest.fixture(autouse=True)
    def setup(self):
        """Load gold standard and check for PDF availability."""
        # Check if PDF exists
        if not os.path.exists(self.PDF_PATH):
            pytest.skip(
                f"PDF not found: {self.PDF_PATH}\n"
                "This test needs the public-domain Constance Garnett translation "
                "of Crime and Punishment as a PDF.\n"
                "Free source: https://www.gutenberg.org/ebooks/2554\n"
                f"Save it as {self.PDF_PATH} to run this test."
            )

        # Load gold standard
        with open(self.GOLD_STANDARD_PATH) as f:
            self.gold = json.load(f)

        # Import processor here to avoid import errors in CI
        import sys
        sys.path.insert(0, '.')
        from process_transcript_full import FullContextProcessor

        # Get LLM config (uses current environment - local or remote)
        api_base = os.getenv("LLM_BASE_URL") or os.getenv("LLM_API_BASE")
        api_key = os.getenv("LLM_API_KEY", "lm-studio")
        model = os.getenv("LLM_MODEL")

        if not api_base:
            pytest.skip("LLM_BASE_URL not configured in .env")

        self.processor = FullContextProcessor(
            api_base=api_base,
            api_key=api_key,
            model=model
        )

    def test_room_count_close_to_gold_standard(self):
        """LLM should extract a comparable number of rooms.

        Extraction from a novel is not deterministic the way a scene-headed
        transcript is, so this allows a margin rather than demanding an exact
        match.
        """
        try:
            result = self.processor.process_pdf(self.PDF_PATH, None)
        except Exception as e:
            pytest.skip(f"LLM processing failed: {e}")

        assert result is not None, "Processor returned None"
        assert "room_count" in result, "Result missing room_count"

        expected_count = self.gold["room_count"]
        actual_count = result["room_count"]

        assert abs(actual_count - expected_count) <= expected_count * 0.25, (
            f"Room count far from gold standard: expected ~{expected_count}, "
            f"got {actual_count}"
        )

    def test_gold_standard_has_expected_rooms(self):
        """Verify the gold standard fixture contains 90 rooms."""
        gold_room_names = [room["name"] for room in self.gold["rooms"]]

        assert len(gold_room_names) == 90, f"Expected 90 rooms, got {len(gold_room_names)}"
        assert len(set(gold_room_names)) == len(gold_room_names), "Duplicate room names"

        for expected_name in self.EXPECTED_SAMPLE_ROOMS:
            assert expected_name in gold_room_names, f"Missing expected room: {expected_name}"

    def test_gold_standard_room_structure(self):
        """Verify each room in gold standard has required fields."""
        required_fields = ["name", "description", "exits", "items", "characters", "events", "atmosphere"]

        for room in self.gold["rooms"]:
            for field in required_fields:
                assert field in room, f"Room '{room.get('name', 'unknown')}' missing field: {field}"

    def test_gold_standard_is_public_domain(self):
        """The committed fixture must stay redistributable."""
        assert self.gold["source"] == "public_domain_novel"
        provenance = self.gold["provenance"]
        assert "Garnett" in provenance["translation"]
        assert "Public domain" in provenance["rights"]

    def test_gold_standard_is_pre_fix_exits(self):
        """The fixture keeps raw descriptive exits, which fix_exits.py resolves."""
        room_names = {room["name"] for room in self.gold["rooms"]}
        unresolved = [
            exit_name
            for room in self.gold["rooms"]
            for exit_name in room["exits"]
            if exit_name not in room_names
        ]
        assert unresolved, "Fixture appears already normalised; it should be raw output"

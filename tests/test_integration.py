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

    These tests verify that an LLM can extract room data from a known
    transcript and produce results comparable to a validated gold standard.

    The gold standard was generated using OpenAI gpt-5.2 (December 2025) and manually verified.
    It contains 28 rooms: 18 Capitol/physical locations and 10 Matrix environments.
    """

    # Expected room names from the gold standard (28 total)
    EXPECTED_ROOM_NAMES = [
        # Capitol/Physical Locations (18)
        "TARDIS - Wooden Console Room",
        "TARDIS - Interior Rooms (Storage/Side Room)",
        "Gallifrey - Sector 7 (Capitol Perimeter / Cloisters Zone)",
        "Communications Tower - Lift (Sector 7)",
        "Communications Tower - Floors (Sector 7)",
        "Records Room (Capitol Archives / Data Retrieval)",
        "Chancellery (High Council Offices)",
        "Capitol Museum",
        "Adytum (Master's Hidden Chamber)",
        "Time Lords Robing Room",
        "Panopticon (Main Chamber)",
        "Panopticon Gallery",
        "Panopticon Antechamber",
        "Detention Room (Capitol Holding Cage)",
        "Capitol Corridor (Evidence Demonstration Corridor)",
        "Service Ducts and Foundations (Below Records Room)",
        "Panopticon Vault",
        "Panopticon - Eye of Harmony Chamber (Under-Dais Mechanism)",
        # Matrix Environments (10)
        "Matrix - Betchworth Quarry",
        "Matrix - Operating Theatre",
        "Matrix - World War One Battlefield",
        "Matrix - Railway Tracks Junction",
        "Matrix - Quarry Valley and Cliff Face (Hunter Pursuit Zone)",
        "Matrix - Lightly Wooded Slope (Biplane Attack)",
        "Matrix - Reed Thicket (Water Trap Area)",
        "Matrix - Poisoned Pool",
        "Matrix - Thicket and Tree Ambush Zone",
        "Matrix - Marsh (Marsh Gas Battleground)",
    ]

    PDF_PATH = "The Doctor Who Transcripts - The Deadly Assassin.pdf"
    GOLD_STANDARD_PATH = "tests/fixtures/deadly_assassin_gold.json"

    @pytest.fixture(autouse=True)
    def setup(self):
        """Load gold standard and check for PDF availability."""
        # Check if PDF exists
        if not os.path.exists(self.PDF_PATH):
            pytest.skip(
                f"PDF not found: {self.PDF_PATH}\n"
                "This test requires the copyrighted Doctor Who transcript.\n"
                "The transcript can be found at: https://www.chakoteya.net/DoctorWho/14-3.html\n"
                "Save it as a PDF to run this test."
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

    def test_room_count_matches_gold_standard(self):
        """LLM should extract the same number of rooms as gold standard."""
        try:
            result = self.processor.process_pdf(self.PDF_PATH, None)
        except Exception as e:
            pytest.skip(f"LLM processing failed: {e}")

        assert result is not None, "Processor returned None"
        assert "room_count" in result, "Result missing room_count"

        expected_count = self.gold["room_count"]
        actual_count = result["room_count"]

        assert actual_count == expected_count, (
            f"Room count mismatch: expected {expected_count}, got {actual_count}"
        )

    def test_gold_standard_has_expected_rooms(self):
        """Verify the gold standard fixture contains all 28 expected rooms."""
        gold_room_names = [room["name"] for room in self.gold["rooms"]]

        assert len(gold_room_names) == 28, f"Expected 28 rooms, got {len(gold_room_names)}"

        for expected_name in self.EXPECTED_ROOM_NAMES:
            assert expected_name in gold_room_names, f"Missing expected room: {expected_name}"

        for actual_name in gold_room_names:
            assert actual_name in self.EXPECTED_ROOM_NAMES, f"Unexpected room in gold standard: {actual_name}"

    def test_gold_standard_room_structure(self):
        """Verify each room in gold standard has required fields."""
        required_fields = ["name", "description", "exits", "items", "characters", "events", "atmosphere"]

        for room in self.gold["rooms"]:
            for field in required_fields:
                assert field in room, f"Room '{room.get('name', 'unknown')}' missing field: {field}"

    def test_gold_standard_matrix_rooms_count(self):
        """Verify gold standard contains exactly 10 Matrix environments."""
        matrix_rooms = [r for r in self.gold["rooms"] if r["name"].startswith("Matrix -")]
        assert len(matrix_rooms) == 10, f"Expected 10 Matrix rooms, got {len(matrix_rooms)}"

    def test_gold_standard_capitol_rooms_count(self):
        """Verify gold standard contains exactly 18 Capitol/physical locations."""
        non_matrix_rooms = [r for r in self.gold["rooms"] if not r["name"].startswith("Matrix -")]
        assert len(non_matrix_rooms) == 18, f"Expected 18 Capitol rooms, got {len(non_matrix_rooms)}"

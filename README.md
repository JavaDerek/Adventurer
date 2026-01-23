# Fiction to Interactive Fiction Converter

[![CI](https://github.com/YOUR_USERNAME/Adventurer/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/Adventurer/actions/workflows/ci.yml)

This tool processes fiction transcripts (in PDF format) and uses an LLM to extract structured room/location data for creating interactive fiction games. Supports both local LLMs and OpenAI API.

## Features

- 📄 Extracts text from PDF transcripts
- 🤖 Supports **local LLMs** (LM Studio, etc.) and **OpenAI API**
- 🔄 Streaming support for large models (handles slow generation gracefully)
- 🏗️ Creates structured JSON with room descriptions, characters, items, and events
- 🔄 Automatically merges duplicate locations
- 💾 Outputs ready-to-use game data

## Requirements

- Python 3.10+
- A local LLM server (LM Studio, vLLM, text-generation-webui, etc.)
- Recommended: **Instruct-tuned model** with 32K+ context window

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Your LLM

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` to configure your LLM(s):

```env
# Local LLM Configuration (for --local flag, default)
LLM_BASE_URL=http://localhost:1234/v1
LLM_MODEL=qwen2.5-32b-instruct
LLM_API_KEY=lm-studio

# OpenAI Configuration (for --remote flag)
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o
OPENAI_BASE_URL=https://api.openai.com/v1

# Generation parameters
MAX_TOKENS=8000
TEMPERATURE=0.0
```

### 3. Recommended Models

For best results, use an **Instruct-tuned** model (not Coder models):

| Model | Context | Notes |
|-------|---------|-------|
| Qwen2.5-32B-Instruct | 32K | Recommended - excellent JSON extraction |
| Qwen2.5-14B-Instruct | 32K | Faster, good quality |
| Mistral-Small-Instruct | 32K | Also works well |

**Avoid Coder models** - they tend to generate code instead of JSON data.

## Usage

### Basic Usage (Local LLM)

```bash
python process_transcript_full.py "transcript.pdf"
```

This creates `transcript_rooms_full.json`

### Use OpenAI API

```bash
python process_transcript_full.py --remote "transcript.pdf"
# or
python process_transcript_full.py -r "transcript.pdf"
```

### Specify Output File

```bash
python process_transcript_full.py input.pdf output.json
python process_transcript_full.py --remote input.pdf output.json
```

### Test Your Connection

```bash
# Test local LLM
python test_connection.py --local

# Test OpenAI API
python test_connection.py --remote
```

## Recommended Workflow

For best results, we recommend a **hybrid approach**:

### 1. Use OpenAI (`--remote`) for Room Extraction

Room extraction is a **one-time setup task** that benefits from maximum accuracy. OpenAI's gpt-4o consistently extracts all locations with high-quality descriptions.

```bash
python process_transcript_full.py --remote "transcript.pdf"
```

**Cost**: Approximately $0.02-0.05 per transcript (a few cents for a full episode).

### 2. Use Local LLM for Runtime Gameplay

Once you have structured room data, the **runtime gameplay engine** can use a local LLM for:
- NPC dialogue generation
- Dynamic event descriptions
- Player action interpretation

Local models are ideal here because:
- Responses need to be fast (< 1 second)
- Tasks are simpler (no complex JSON extraction)
- Volume is high (many calls per play session)

### Why This Split?

| Task | Best Choice | Reason |
|------|-------------|--------|
| Room extraction | OpenAI | One-time, accuracy-critical, complex JSON |
| Gameplay runtime | Local LLM | Fast, high-volume, simpler prompts |

### Testing Your Local LLM

The gold standard integration test benchmarks your local LLM against OpenAI's output:

```bash
pytest tests/test_integration.py::TestGoldStandardComparison -v
```

If your local hardware can match the gold standard (20 rooms for the test transcript), you can use `--local` for everything!

## Output Format

The script generates JSON in this format:

```json
{
  "title": "The Deadly Assassin",
  "format": "interactive_fiction_v1",
  "room_count": 18,
  "rooms": [
    {
      "name": "TARDIS Console Room",
      "description": "The central control room of the TARDIS with a hexagonal console...",
      "exits": ["Gallifrey - sector 7"],
      "items": ["Console", "Viewscreen", "Time Rotor"],
      "characters": ["The Doctor"],
      "events": ["Doctor receives vision of the Panopticon"],
      "atmosphere": "mysterious"
    },
    {
      "name": "Matrix - Cliff Face",
      "description": "A dangerous cliff within the Matrix simulation...",
      "exits": ["Matrix - Valley"],
      "items": ["rope", "rocks"],
      "characters": ["The Doctor", "Goth"],
      "events": ["Doctor climbs to escape pursuit"],
      "atmosphere": "perilous"
    }
  ],
  "metadata": {
    "all_characters": ["The Doctor", "Spandrell", "Goth"],
    "all_locations": ["TARDIS Console Room", "Matrix - Cliff Face"]
  }
}
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run unit tests only (mocked, no LLM calls - what CI runs)
python -m pytest tests/ -v -m "not integration"

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=term-missing -m "not integration"

# Run integration tests (requires LLM access)
python -m pytest tests/ -v -m integration
```

### Project Structure

```
├── process_transcript_full.py  # Main processor (full context + streaming)
├── test_connection.py          # LLM connection test utility
├── tests/
│   ├── test_processor.py       # Unit tests (mocked, no LLM calls)
│   ├── test_integration.py     # Integration tests (real LLM calls)
│   └── conftest.py             # Pytest configuration
├── .env.example                # Example configuration
├── .coveragerc                 # Coverage configuration
├── pytest.ini                  # Pytest markers configuration
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development dependencies
└── .github/workflows/ci.yml    # GitHub Actions CI
```

## Troubleshooting

### "Connection refused" error
- Make sure your local LLM server is running
- Check the `LLM_BASE_URL` URL matches your server
- Test with: `curl http://localhost:1234/v1/models`

### Request timeout
- The script uses **streaming mode** to handle slow models
- If still timing out, check your LLM server isn't overloaded

### Model returns code instead of JSON
- Switch from Coder model to Instruct model
- Example: Use `qwen2.5-32b-instruct` instead of `qwen2.5-coder-32b`

### Poor quality extractions
- Use a larger, more capable model
- Ensure model has sufficient context window (32K+ recommended)
- Try adjusting temperature (0.0 for deterministic, 0.3-0.5 for variety)

## License

MIT License - Feel free to use and modify!

## Contributing

Found a bug or want to improve the prompts? Pull requests welcome!

# Fiction to Interactive Fiction Converter

[![CI](https://github.com/JavaDerek/Adventurer/actions/workflows/ci.yml/badge.svg)](https://github.com/JavaDerek/Adventurer/actions/workflows/ci.yml)

This tool processes fiction transcripts (novels, TV scripts, plays in PDF format) and uses an LLM to extract structured room/location data for creating interactive fiction games. The complete pipeline takes you from PDF to a playable game served by
[run-dmcp](https://github.com/JavaDerek/run-dmcp).

## Features

- 📄 Extracts text from PDF transcripts with intelligent cleaning (handles encoding issues)
- 🤖 Supports **local LLMs** (Ollama, LM Studio, vLLM, etc.) and **OpenAI API**
- 📚 Smart chunking for large documents (splits at chapter/part boundaries)
- 🔄 Streaming support for large models (handles slow generation gracefully)
- 🏗️ Creates structured JSON with room descriptions, characters, items, and events
- 🔄 Automatically merges duplicate locations
- 🗺️ Map analysis to identify connectivity issues
- 🔧 LLM-powered map healing to fix broken connections
- 🎮 Direct loading into run-dmcp for gameplay in any MCP client

## Requirements

- Python 3.10+
- A local LLM server (Ollama, LM Studio, vLLM, text-generation-webui, etc.) OR OpenAI API key
- Recommended: **Instruct-tuned model** with 32K+ context window
- For gameplay: [run-dmcp](https://github.com/JavaDerek/run-dmcp) built (Node 20+)

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
# Local LLM Configuration (for --local flag)
LLM_BASE_URL=http://localhost:1234/v1
LLM_MODEL=qwen2.5-32b-instruct
LLM_API_KEY=lm-studio

# OpenAI Configuration (for --remote flag, default for processing)
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o
OPENAI_BASE_URL=https://api.openai.com/v1

# Generation parameters
MAX_TOKENS=16000
TEMPERATURE=0.0

# Context window limit for chunking large documents
# Set lower than model's actual limit to leave room for prompts
CONTEXT_TOKEN_LIMIT=25000
```

### 3. Recommended Models

**For room extraction, OpenAI models are strongly recommended.** Even GPT-5.2 produces exits as descriptive text that requires post-processing with `fix_exits.py`. Local models will likely produce lower quality output requiring more manual cleanup.

| Model | Context | Extraction Quality | Notes |
|-------|---------|-------------------|-------|
| GPT-5.2 | 128K | Best | Recommended for complex novels |
| GPT-4o | 128K | Excellent | Good balance of cost/quality |
| Qwen2.5-32B-Instruct | 32K | Moderate | May work for simple scripts |
| Qwen2.5-14B-Instruct | 32K | Limited | Short transcripts only |

**Local models** may be adequate for:
- Short scripts (< 50 rooms)
- Simple narratives with clear location names
- Situations where you'll do significant manual editing anyway

**Avoid Coder models** - they tend to generate code instead of JSON data.

## Complete Workflow

The full pipeline from PDF to playable game:

```
1. PDF Input
   ↓
2. process_transcript_full.py  →  rooms_full.json (raw extraction)
   ↓
3. fix_exits.py                →  rooms_fixed.json (normalized exits)
   ↓
4. load_to_run_dmcp.py         →  Playable game in run-dmcp
```

**Optional**: Run `analyze_map.py` at any stage to inspect connectivity issues.

## Usage

### Step 1: Extract Rooms from PDF

```bash
# Use OpenAI (recommended for best accuracy)
python process_transcript_full.py --remote "novel.pdf"

# Use local LLM
python process_transcript_full.py --local "novel.pdf"

# Specify output file
python process_transcript_full.py --remote input.pdf output.json
```

This creates `novel_rooms_full.json` with all extracted locations.

**For large documents** (novels, long scripts), the processor automatically:
- Detects chapter/part markers
- Splits the document at natural boundaries
- Processes each chunk with context from previous chunks
- Deduplicates and merges the results

### Step 2: Analyze Map Connectivity (Optional)

```bash
python analyze_map.py rooms_full.json
```

This identifies:
- Broken references (exits pointing to non-existent rooms)
- One-way doors (can go A→B but not B→A)
- Unreachable rooms
- Disconnected subgraphs

### Step 3: Fix Exit References

The LLM often extracts exits as descriptive text ("Down toward the river") instead of room names. Fix this with:

```bash
python fix_exits.py rooms_full.json --connect-subgraphs
```

Options:
- `--connect-subgraphs`: Connect isolated room clusters
- `--min-score 0.7`: Fuzzy match threshold (0-1)
- `--no-bidirectional`: Don't add reverse connections
- `-o output.json`: Specify output file

This creates `rooms_full_fixed.json` with valid room connections.

### Step 4: Load into run-dmcp for Gameplay

First, build run-dmcp once:

```bash
git clone https://github.com/JavaDerek/run-dmcp.git
cd run-dmcp && npm ci && cd client && npm ci && cd ..
npm run build
```

Then load your game:

```bash
python load_to_run_dmcp.py rooms_fixed.json --server-path ~/run-dmcp
```

The loader speaks MCP over stdio, creates every entity, and prints a game ID
plus a web UI link:

```
Game ID: ddad8ad6-b9b2-4543-846c-96e868a4248d
Locations: 28
Connections: 22
Characters: 25
Items: 114
Notes: 28 (90 beats)

Web UI: http://localhost:3456/games/ddad8ad6-b9b2-4543-846c-96e868a4248d
```

`--server-path` defaults to `$RUN_DMCP_PATH`, then `~/run-dmcp`.

To play, point an MCP client at run-dmcp and use `load_game <game-id>`. For
Claude Desktop, add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "run-dmcp": {
      "command": "node",
      "args": ["/absolute/path/to/run-dmcp/dist/index.js"]
    }
  }
}
```

How the JSON maps onto run-dmcp's tools — and the decisions behind that mapping
— is documented in [docs/RUN_DMCP_INTEGRATION.md](docs/RUN_DMCP_INTEGRATION.md).

### Alternative: LLM-Powered Map Healing

For complex connectivity issues, you can use the LLM-powered healer:

```bash
# Analyze only (no changes)
python heal_map.py rooms_full.json --analyze-only

# Heal with OpenAI (default)
python heal_map.py rooms_full.json

# Heal with local LLM
python heal_map.py --local rooms_full.json
```

**Note**: For maps with many rooms (100+), `fix_exits.py` is faster and more reliable than `heal_map.py`.

### Test Your Connection

```bash
# Test local LLM
python test_connection.py --local

# Test OpenAI API
python test_connection.py --remote
```

## Recommended Workflow

### Expect Post-Processing

Even with the best models, LLM-extracted room data typically requires cleanup:
- Exits are often descriptive ("Down toward the river") rather than room names
- Complex novels may produce hundreds of disconnected room clusters
- Character and item names may have inconsistencies

The `fix_exits.py` script handles most of these issues automatically.

### Use OpenAI (`--remote`) for Room Extraction

Room extraction is a **one-time setup task** where quality matters. OpenAI's models produce better structured output that requires less manual cleanup.

```bash
python process_transcript_full.py --remote "transcript.pdf"
python fix_exits.py --connect-subgraphs transcript_rooms_full.json
python load_to_run_dmcp.py transcript_rooms_full_fixed.json
```

**Cost**: Approximately $0.02-0.10 per transcript depending on length.

### Gameplay

Once loaded, run-dmcp owns the world state and your MCP client narrates over it
— generating NPC dialogue, describing events, and interpreting player actions.

| Task | Tool | Notes |
|------|------|-------|
| Room extraction | OpenAI API or local LLM | One-time, quality matters |
| Exit fixing | fix_exits.py | Deterministic and reproducible, no API cost |
| Game loading | load_to_run_dmcp.py | Creates run-dmcp entities |
| Gameplay | MCP client + run-dmcp | Interactive play |

## Output Format

The processor generates JSON in this format:

```json
{
  "title": "Crime_and_Punishment",
  "format": "interactive_fiction_v1",
  "source": "fiction",
  "processing_method": "chunked",
  "room_count": 90,
  "rooms": [
    {
      "name": "Raskolnikov's garret",
      "description": "A cramped attic room where Raskolnikov has been ill and delirious...",
      "exits": ["Staircase", "Street outside the lodging house"],
      "items": ["Parcel of new clothes", "Money", "Table"],
      "characters": ["Raskolnikov"],
      "events": ["Raskolnikov leaves his garret, trying to avoid his landlady"],
      "atmosphere": "Constricted and anxious, marked by secrecy and dread"
    }
  ],
  "metadata": {
    "all_characters": ["Raskolnikov", "Sonia", "Porfiry Petrovitch", "..."],
    "all_locations": ["Raskolnikov's garret", "Hay Market", "Police station", "..."]
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
├── process_transcript_full.py  # Main PDF→JSON processor
├── analyze_map.py              # Map connectivity analyzer
├── fix_exits.py                # Exit text normalizer (fuzzy matching)
├── heal_map.py                 # LLM-powered map healer
├── load_to_run_dmcp.py         # run-dmcp game loader
├── test_connection.py          # LLM connection test utility
├── tests/
│   ├── test_processor.py       # Processor unit tests
│   ├── test_analyzer.py        # Map analyzer tests
│   ├── test_healer.py          # Map healer tests
│   ├── test_loader.py          # run-dmcp loader tests
│   ├── test_integration.py     # Integration tests (real LLM)
│   ├── conftest.py             # Pytest configuration
│   └── fixtures/               # Test fixtures
├── docs/                       # Architecture and integration notes
├── CLAUDE.md                   # Orientation for AI coding agents
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
- Check the `LLM_BASE_URL` matches your server — the port differs by backend
  (LM Studio defaults to `1234`, Ollama to `11434`)
- Test with: `curl $LLM_BASE_URL/models`

### Request timeout
- The script uses **streaming mode** to handle slow models
- If still timing out, check your LLM server isn't overloaded
- Increase timeout in script if needed

### Model returns code instead of JSON
- Switch from Coder model to Instruct model
- Example: Use `qwen2.5-32b-instruct` instead of `qwen2.5-coder-32b`

### Poor quality extractions
- Use a larger, more capable model
- Ensure model has sufficient context window (32K+ recommended)
- Try adjusting temperature (0.0 for deterministic, 0.3-0.5 for variety)

### PDF text has garbled characters
- The processor automatically detects and fixes common encoding issues
- If spaces appear as "1" characters, this is handled automatically
- For other encoding issues, try a different PDF source

### Large document fails or runs out of context
- Set `CONTEXT_TOKEN_LIMIT` in `.env` to a value lower than your model's limit
- The processor will automatically chunk at chapter boundaries
- Each chunk is processed separately and results are merged

### Many broken references in output
- Run `fix_exits.py --connect-subgraphs` to normalize exits
- This matches descriptive exits to actual room names
- Disconnected areas will be linked together

### Load finishes but "Characters: 0"

Older run-dmcp builds declared an output schema for the character tools that
omitted two fields they actually return, so a validating MCP client rejected
every result ([run-dmcp#24](https://github.com/JavaDerek/run-dmcp/issues/24),
since fixed). Pull and rebuild run-dmcp. The loader detects this and continues
anyway, logging a warning that says so. Details in
[docs/RUN_DMCP_INTEGRATION.md](docs/RUN_DMCP_INTEGRATION.md).

### run-dmcp ignores DMCP_DB_PATH or DMCP_HTTP_PORT

The MCP SDK launches the server with a scrubbed environment. The loader forwards
`os.environ` explicitly; if you're calling the MCP SDK yourself, pass
`env=dict(os.environ)` to `StdioServerParameters` or the server will silently
use its default database.

### "run-dmcp not found at .../dist/index.js"

run-dmcp needs building before first use:

```bash
cd ~/run-dmcp && npm ci && cd client && npm ci && cd ..
npm run build
```

### Rooms have fewer exits than the loader reported

Exit directions must be unique within a room — run-dmcp replaces any existing
exit sharing a direction. The loader labels exits `toward <destination>` to
guarantee uniqueness. A custom labelling scheme that repeats will lose exits.

### heal_map.py fails with "Could not extract valid JSON"
- The map may be too large for a single LLM call
- Use `fix_exits.py` instead for large maps (100+ rooms)
- Or process smaller subsets manually

## Documentation

| Document | Contents |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Pipeline stages, what each script owns, where state lives |
| [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) | The room JSON contract every stage reads and writes |
| [docs/RUN_DMCP_INTEGRATION.md](docs/RUN_DMCP_INTEGRATION.md) | Tool-by-tool mapping onto run-dmcp, and why each call is shaped that way |
| [CLAUDE.md](CLAUDE.md) | Orientation for AI coding agents working in this repo |

## License

MIT License - Feel free to use and modify!

## Contributing

Found a bug or want to improve the prompts? Pull requests welcome!
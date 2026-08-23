# Architecture

Adventurer is a batch pipeline, not a service. Each stage is a standalone CLI
script that reads a file and writes a file. There is no shared runtime, no
daemon, and no database of its own — the only persistent state lives in
run-dmcp once a game is loaded.

```
   PDF
    │
    ▼
process_transcript_full.py ──► <name>_rooms_full.json      (raw extraction, LLM)
    │
    ▼
fix_exits.py ───────────────► <name>_rooms_full_fixed.json (normalised, deterministic)
    │
    ▼
load_to_run_dmcp.py ────────► run-dmcp SQLite database     (MCP over stdio)

analyze_map.py ─────────────► report only, runs at any stage
heal_map.py ────────────────► alternative to fix_exits.py, LLM-powered
```

## What each script owns

| Script | Input | Output | LLM? |
|---|---|---|---|
| `process_transcript_full.py` | PDF | rooms JSON | Yes — the only expensive stage |
| `analyze_map.py` | rooms JSON | printed report | No |
| `fix_exits.py` | rooms JSON | rooms JSON | No — fuzzy string matching |
| `heal_map.py` | rooms JSON | rooms JSON | Yes |
| `load_to_run_dmcp.py` | rooms JSON | run-dmcp entities | No |
| `test_connection.py` | — | printed report | Yes (a trivial ping) |

## Design properties worth preserving

**Every stage is resumable by hand.** The intermediate artefact is a plain JSON
file you can open, edit, and feed to the next stage. When an extraction comes
out wrong, fixing the JSON is usually faster than re-running the LLM. Keep it
that way: don't collapse stages into one command.

**Only two stages cost money or need a GPU.** Extraction and healing call an
LLM; everything else is deterministic — genuinely so, and tested. `fix_exits.py`
and `analyze_map.py` once picked representative rooms out of Python sets, whose
iteration order follows per-process randomised string hashes, so the same input
produced a different map on each run. Both now pick with `min()`, and
`tests/test_fix_exits.py` runs the repair chain under four `PYTHONHASHSEED`
values and fails if the outputs differ. Never reintroduce `next(iter(a_set))`
or `list(a_set)[0]` on a path that shapes output. `fix_exits.py` exists because it solves
most connectivity problems without a model, and it is the recommended path for
maps over ~100 rooms — `heal_map.py` struggles to hold a large map in one call.

**The JSON is the authority on the map.** The loader transcribes connectivity
exactly as the JSON states it, including one-way doors. It does not repair the
map on the way in. If the map is wrong, fix it in the JSON where the change is
visible and re-runnable — see
[RUN_DMCP_INTEGRATION.md](RUN_DMCP_INTEGRATION.md) decision 2.

**Extraction quality varies by source material.** A script with named scene
headings extracts cleanly. A novel with fluid, unnamed settings produces many
near-duplicate rooms that need merging — Crime and Punishment yields 90 rooms
with only 2 of 206 exits resolving to a real room name before `fix_exits.py`
runs. Expect novels to need the normalisation pass; expect scripts not to.

## Chunking

Documents that exceed `CONTEXT_TOKEN_LIMIT` are split at chapter or part
boundaries, processed with context carried from the previous chunk, then merged
with duplicate locations combined. This is why room counts can exceed what the
narrative "really" contains — the merge is heuristic.

## Testing

Unit tests mock every external dependency; no test calls an LLM or starts
run-dmcp. Integration tests are marked `integration` and deselected in CI.

```bash
python -m pytest tests/ -v -m "not integration"
python -m pytest tests/ --cov=. --cov-report=term-missing --cov-fail-under=80 -m "not integration"
```

The loader's tests assert the *shape of the MCP calls* against a mocked
session, because that shape is the contract with run-dmcp and it has broken
once already. See [RUN_DMCP_INTEGRATION.md](RUN_DMCP_INTEGRATION.md).

A mocked session cannot catch a server that rejects a call it declared it would
accept. Both bugs found during the run-dmcp port were of exactly that kind, and
both were invisible to the unit tests. After changing the loader, do one real
load against a scratch database:

```bash
python fix_exits.py tests/fixtures/crime_and_punishment_gold.json \
  --connect-subgraphs -o /tmp/cp_fixed.json
DMCP_DB_PATH=/tmp/scratch.db python load_to_run_dmcp.py \
  /tmp/cp_fixed.json --server-path ~/run-dmcp
```

Expect 90 locations, 228 connections, 113 characters, 191 items, 90 notes.
Those numbers are reproducible: the repair stages are deterministic, and
`tests/test_fix_exits.py` fails if that stops being true.
Load the fixture directly, without `fix_exits.py`, and connections drop to 2 —
that is correct behaviour, not a bug.

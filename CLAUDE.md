# CLAUDE.md

Orientation for AI coding agents working in this repository.

## What this is

Adventurer converts fiction (novels, TV transcripts, plays) in PDF form into
interactive fiction playable through [run-dmcp](https://github.com/JavaDerek/run-dmcp),
an MCP server that owns game state.

It is a **batch pipeline of standalone CLI scripts**, not a service. No daemon,
no framework, no package layout — top-level `.py` files that read a file and
write a file. Don't restructure it into a package; the flat layout is what makes
each stage independently runnable and hand-editable.

Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) before making structural
changes, and [docs/RUN_DMCP_INTEGRATION.md](docs/RUN_DMCP_INTEGRATION.md) before
touching `load_to_run_dmcp.py`.

## Setup

```bash
pip install -r requirements-dev.txt   # includes requirements.txt
cp .env.example .env                  # configure your LLM
```

## Commands

```bash
# What CI runs
python -m pytest tests/ -v --tb=short -m "not integration"
python -m pytest tests/ --cov=. --cov-report=term-missing --cov-fail-under=80 -m "not integration"

# Integration tests (real LLM; not run in CI)
python -m pytest tests/ -v -m integration

ruff check . --fix
```

CI also `py_compile`s every top-level script. **If you add or rename a
top-level script, update `.github/workflows/ci.yml`** — it lists them by name.

## Non-negotiables

**Tests first.** Write or update a test so it fails against current code, then
make it pass. This applies to bug fixes and behaviour changes, not just
features.

**Coverage floor is 80%** and CI enforces it.

**Unit tests never touch the network, an LLM, or a real run-dmcp.** Mock
everything. Anything that needs a live service is marked `integration` and
deselected in CI.

**Log before swallowing.** Every `except` that returns or continues instead of
re-raising calls `logger.exception()` with context. Silent `except: pass` is a
bug — there was one in the original loader and it hid failures for the entire
load.

**No synthetic data.** Never fabricate names, URLs, or contact details. Empty
beats invented.

## Testing against run-dmcp

The loader's unit tests assert the shape of MCP calls against a mocked session.
That catches parameter drift but **cannot** catch a server rejecting a call it
declared it would accept. Both bugs found during the run-dmcp port were of that
kind and both passed the mocked tests.

So after changing `load_to_run_dmcp.py`, do one real load into a scratch
database:

```bash
# The fixture is raw extraction output, so normalise its exits first --
# loading it directly yields only 2 connections and barely exercises that path.
python fix_exits.py tests/fixtures/crime_and_punishment_gold.json \
  --connect-subgraphs -o /tmp/cp_fixed.json

DMCP_DB_PATH=/tmp/scratch.db DMCP_HTTP_PORT=39456 \
  python load_to_run_dmcp.py /tmp/cp_fixed.json --server-path ~/run-dmcp
```

Expect: 90 locations, 228 connections, 113 characters, 191 items, 90 notes
(153 beats), 0 warnings. A load reporting `Characters: 0` or `Connections: 0`
while other counts look fine means a schema mismatch, not empty input.

**Always set `DMCP_DB_PATH` to a scratch file.** Without it run-dmcp writes to
its default library and mixes your test into real games.

For a scale check, the 568-room Master and Margarita file is the stress case:
1436 connections, 602 characters, 1300 items, and it should finish with zero
warnings.

## Traps that have already cost time

**run-dmcp's tool surface is not DMCP's.** Same tool names, different
parameters. `connect_locations` renamed every argument and made directions
required; `create_item` swapped `locationId` for `ownerId`/`ownerType`;
`add_event` is gone. Check the mapping table in
[docs/RUN_DMCP_INTEGRATION.md](docs/RUN_DMCP_INTEGRATION.md) rather than
assuming a name that exists behaves the same.

**Exit directions must be unique within a room.** run-dmcp deletes any existing
exit sharing a direction with a new one, so a constant label silently reduces
every room to one exit. Verify with a stored-exit count after a real load.

**`stdio_client` scrubs the child environment.** Pass `env=dict(os.environ)` in
`StdioServerParameters` or the server never sees `DMCP_*` config, and fails
silently by using defaults.

**Monkeypatching SDK internals rots.** The original loader patched
`_validate_tool_result`, which the MCP SDK renamed to the public
`validate_tool_result` in v2.0. The patch then did nothing — it just added an
unused attribute, no error. If you must patch, assert the patch landed.

**`room["name"]` is the primary key.** Exits reference rooms by exact name
string. Truncating or normalising a name for display must not change the key
used for lookups.

**Never pick from a set on an output path.** `next(iter(a_set))` and
`list(a_set)[0]` follow randomised string hashes, so they differ between
processes. That made `fix_exits.py` emit a different map on every run of
identical input — while looking perfectly deterministic in-process, which is
why unit tests missed it for so long. Use `min()` or `sorted()`, and note that
a dict built by iterating a set inherits the same problem through its insertion
order. `tests/test_fix_exits.py` guards this across four `PYTHONHASHSEED`
values.

## Repository conventions

- Every module gets `logger = logging.getLogger(__name__)`.
- CLI entry points call `logging.basicConfig(..., stream=sys.stderr)` in
  `main()`; keep diagnostics off stdout.
- Limits mirrored from run-dmcp (`NAME_MAX`, `DESCRIPTION_MAX`, `CONTENT_MAX`)
  live at the top of `load_to_run_dmcp.py`. If run-dmcp changes them, change
  them here and in `docs/RUN_DMCP_INTEGRATION.md` in the same commit.
- `ruff` has no config here and the repo carries a pre-existing lint backlog in
  the older scripts. Leave files you aren't touching alone; keep files you do
  touch clean.

## Known upstream issue

[run-dmcp#24](https://github.com/JavaDerek/run-dmcp/issues/24) — the character
tools declared an output schema omitting `voice` and `imageGen`, so a
validating client rejected every character. **Fixed upstream.** Against a
current build, validation stays on and passes.

`call_tool()` still tolerates it for anyone on an older checkout, but lazily:
validation stays enabled, and only a server that actually violates its own
schema triggers a single explanatory warning plus a retry. Do not turn this
into an unconditional disable — that would hide the next instance of exactly
this bug.

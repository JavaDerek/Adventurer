# run-dmcp Integration Contract

How [`load_to_run_dmcp.py`](../load_to_run_dmcp.py) talks to
[run-dmcp](https://github.com/JavaDerek/run-dmcp), and why each call is shaped
the way it is.

This file exists so nobody has to re-read run-dmcp's TypeScript to change the
loader. If run-dmcp's tool surface moves, update this table in the same commit.

## Why this document exists

Adventurer originally targeted [DMCP](https://github.com/shawnrushefsky/dmcp),
which is retired. run-dmcp continues that codebase, but the tool surface drifted:
same names, different parameters, and one tool gone entirely. The port was not a
rename — every call below changed except `create_game`.

## Tool mapping

| Stage | DMCP (old) | run-dmcp (current) | What changed |
|---|---|---|---|
| Game | `create_game` | `create_game` | Unchanged: `name`, `setting`, `style`. |
| Rooms | `create_location` | `create_location` | Gained `properties.atmosphere`. |
| Exits | `connect_locations` | `connect_locations` | **Breaking.** New parameter names, directions now required, no `gameId`. |
| People | `create_character` | `create_character` | Compatible, but see *Output-schema mismatch*. |
| Things | `create_item` | `create_item` | **Breaking.** `locationId` → `ownerId` + `ownerType`; `description` moved into `properties`. |
| Beats | `add_event` | *(removed)* | Mapped to `create_note`, not `log_event`. See below. |

## The three decisions worth knowing

### 1. Exit directions are synthesised, and must be unique per room

run-dmcp requires `fromDirection` and `toDirection`. Adventurer's JSON has no
compass data — an exit is just a destination room name.

The important detail is in run-dmcp's `connectLocations()`: before adding an
exit it **removes any existing exit with the same direction**.

```ts
fromLocation.properties.exits = fromLocation.properties.exits.filter(
  (e) => e.direction !== params.fromDirection
);
```

So a constant label like `"onward"` would leave every room with exactly one
exit, silently discarding the rest. The loader uses `toward <destination>`
(see `exit_direction()`), which is unique per room because destination names are
unique, and which reads naturally when a model narrates it.

### 2. Connections are created with `bidirectional: False`

run-dmcp defaults `bidirectional` to **true** (`params.bidirectional !== false`).
That default is wrong for this pipeline:

- `fix_exits.py` already writes the reverse exit into the JSON whenever one
  should exist, so letting the server add it too would double up.
- `analyze_map.py` reports one-way doors as a finding. If the loader quietly
  repaired them on import, that analysis would describe a map that never
  reaches the server.

The JSON is the authority on connectivity. The loader transcribes it exactly.

### 3. Source beats become notes, not events

Each room carries an `events` list — things that happen *in the source text*.
The obvious target is `log_event`, but that writes **narrative history**: it is
the record of what has happened in this playthrough. Seeding 90 unplayed beats
there tells the narrator they already occurred.

`create_note` is the honest home for them:

```python
{"gameId": ..., "title": "Source beats: <room>", "content": "- beat\n- beat",
 "category": "plot", "relatedEntityId": <locationId>,
 "relatedEntityType": "location", "tags": ["adventurer", "source-beat"]}
```

Available to the narrator as reference material, attached to the room, and not
asserted as history. One note per room, not one per beat.

A correct import leaves `narrative_events` empty. That is a useful smoke test.

## Two client-side workarounds

### Output-schema mismatch on `create_character`

run-dmcp declares an `outputSchema` for `create_character`
(`src/utils/output-schemas.ts`, `characterOutputSchema`) that lists ten fields.
`createCharacter` actually returns two more — `voice` and `imageGen`. The schema
is generated with `additionalProperties: false`, so a spec-compliant MCP client
rejects **every** `create_character` result:

```
Invalid structured content returned by tool create_character:
Additional properties are not allowed ('imageGen', 'voice' were unexpected)
```

The symptom is a load that reports `Characters: 0` while everything else
succeeds.

**This was an upstream bug** — [run-dmcp#24](https://github.com/JavaDerek/run-dmcp/issues/24),
**fixed** by adding `voice` and `imageGen` to `characterOutputSchema`. It
affected all four tools declaring `outputSchema: characterOutputSchema`:
`create_character`, `get_character`, `update_character` and
`get_character_by_name`. Against a current run-dmcp build, output validation
stays on and passes.

The loader still tolerates it, because someone may be running an older
checkout. Validation is **not** disabled up front — doing that would suppress
genuine schema drift in some future run-dmcp, which is exactly the class of bug
this was. Instead `call_tool()` leaves validation enabled and reacts only if a
server actually returns content its own schema rejects: it logs one warning
naming the cause and the remedy, disables validation for the rest of that load,
and retries. A healthy server never triggers it.

Note the hook name: the MCP Python SDK renamed `_validate_tool_result` to the
public `validate_tool_result` in v2.0. Adventurer's original code patched only
the private name, so after the SDK upgrade the patch became a no-op that
silently added an unused attribute. The loader now patches whichever name
exists and logs a warning if neither does.

### The child process gets a scrubbed environment

`stdio_client` does **not** inherit the parent environment unless you pass one.
Without `env=dict(os.environ)` in `StdioServerParameters`, run-dmcp never sees
`DMCP_DB_PATH`, `DMCP_HTTP_PORT` or `DMCP_LOG_LEVEL` — so it ignores the
database you pointed it at and writes to its default instead. That failure is
silent and easy to miss, because the load itself succeeds.

## Field limits

Mirrored from run-dmcp's `src/utils/validation.ts`. Over-long values are
rejected by the server's zod schemas, which would abort a load partway through,
so the loader clamps locally (`truncate()`).

| Constant | Value | Applies to |
|---|---|---|
| `NAME_MAX` | 200 | game/location/character/item names, note titles, exit directions |
| `DESCRIPTION_MAX` | 5000 | descriptions, atmosphere |
| `CONTENT_MAX` | 50000 | note content |

Real extractions sit well inside these (the largest observed room name is 92
characters, the largest description 613), so clamping is a guard against odd
extractions rather than a routine occurrence.

## Environment variables run-dmcp reads

| Variable | Default | Purpose |
|---|---|---|
| `DMCP_DB_PATH` | `~/.local/share/dmcp/games.db`, else `./data/games.db` | SQLite database |
| `DMCP_HTTP_PORT` | `3456` | Web UI port |
| `DMCP_LOG_LEVEL` | `info` | `debug` / `info` / `warn` / `error` |

Point `DMCP_DB_PATH` at a scratch file when testing so you don't mix
experiments into a real game library.

## Verifying a load by hand

```bash
sqlite3 "$DMCP_DB_PATH" "SELECT COUNT(*) FROM locations;"
sqlite3 "$DMCP_DB_PATH" "SELECT COUNT(*) FROM narrative_events;"  -- expect 0
sqlite3 "$DMCP_DB_PATH" "SELECT name, json_extract(properties,'$.exits') FROM locations LIMIT 5;"
```

If stored exits are fewer than the connection count the loader reported, the
direction labels are colliding — see decision 1.

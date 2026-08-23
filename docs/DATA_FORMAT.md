# Room JSON Format

The contract every pipeline stage reads and writes. `format` is
`interactive_fiction_v1`.

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
      "description": "A cramped attic room where Raskolnikov has been ill...",
      "exits": ["Staircase", "Street outside the lodging house"],
      "items": ["Parcel of new clothes", "Money", "Table"],
      "characters": ["Raskolnikov"],
      "events": ["Raskolnikov leaves his garret, avoiding his landlady"],
      "atmosphere": "Constricted and anxious, marked by secrecy and dread",
      "notes": "Optional; added by heal_map.py"
    }
  ],
  "metadata": {
    "all_characters": ["Raskolnikov", "Sonia", "..."],
    "all_locations": ["Raskolnikov's garret", "Hay Market", "..."]
  }
}
```

## Room fields

| Field | Required | Notes |
|---|---|---|
| `name` | Yes | **The primary key.** Must be unique; `exits` reference rooms by this exact string. |
| `description` | Yes | Prose. Becomes the run-dmcp location description. |
| `exits` | Yes | Array of destination **room names**, not directions. May be empty. |
| `items` | No | Array of item names. Loaded as location-owned items. |
| `characters` | No | Array of names; parentheticals are stripped on load. |
| `events` | No | Beats from the source text. Loaded as GM notes, not history. |
| `atmosphere` | No | One line of tone. Maps to run-dmcp's `properties.atmosphere`. |
| `notes` | No | Provenance written by `heal_map.py`. Not loaded. |

## The two things that trip people up

### `exits` holds room names, not directions

There is no compass data anywhere in this format. An exit is a destination name
that must match another room's `name` exactly. The loader synthesises direction
labels when it reaches run-dmcp, which does require them.

Fresh LLM output frequently violates this: models emit descriptive exits like
`"Down toward the river"` instead of a room name. That is what `fix_exits.py`
repairs, by fuzzy-matching descriptive text to real room names.

### `exits` is directional and asymmetric on purpose

`A` listing `B` does not imply `B` lists `A`. One-way doors are real and
`analyze_map.py` reports them. `fix_exits.py` adds the reverse edge by default
(disable with `--no-bidirectional`), which is why a fixed file usually has both
sides — but the format does not require it, and the loader does not assume it.

## Character names

Names carry parenthetical qualifiers from transcripts:
`"Porfiry (via the doorway)"`, `"Svidrigailov (as ghost/voice)"`,
`"Sonia (briefly)"`. `normalize_character_name()` strips these and collapses
whitespace, so all three variants of one person become a single character. A
character is placed at the first room it appears in.

A character literally named `Player` is created as the player character.

## Duplicate and near-duplicate rooms

Chunked processing merges rooms whose names match after normalisation, but
near-duplicates survive — a novel can yield several rooms for what a reader
would call one place. This is inherent to extracting locations from prose that
never names them consistently. Merging further is a manual edit.

## Field limits

The loader clamps to run-dmcp's schema limits (names 200, descriptions 5000,
note content 50000 characters) rather than letting the server reject a value
and abort a partial load. Real extractions sit well inside these.

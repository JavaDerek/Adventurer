#!/usr/bin/env python3
"""
Fiction to Interactive Fiction Processor (Full Context Version)

Converts narrative fiction (novels, TV scripts, plays) into structured
room/location data for interactive fiction games.
Uses the full 128K context window to process entire documents at once.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
import PyPDF2
from openai import OpenAI
from dotenv import load_dotenv
import httpx

# Load environment variables from .env file
load_dotenv()

# Configuration
LLM_API_BASE = os.getenv("LLM_BASE_URL") or os.getenv("LLM_API_BASE", "http://localhost:8000/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "lm-studio")
LLM_MODEL = os.getenv("LLM_MODEL", "local-model")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))  # Increased for full document processing
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

# Chunking configuration for large documents
CONTEXT_TOKEN_LIMIT = int(os.getenv("CONTEXT_TOKEN_LIMIT", "200000"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "500"))
MIN_CHUNK_TOKENS = int(os.getenv("MIN_CHUNK_TOKENS", "10000"))


class FullContextProcessor:
    """Process entire fiction documents (novels, scripts, plays) in one pass with full LLM context"""

    def __init__(self, api_base: str, api_key: str, model: str):
        self.client = OpenAI(
            base_url=api_base,
            api_key=api_key,
            timeout=httpx.Timeout(600.0, connect=60.0)  # 10 min read, 1 min connect
        )
        self.model = model

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF file"""
        print(f"📄 Extracting text from {pdf_path}...")

        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text_content = []

            for page_num, page in enumerate(pdf_reader.pages, 1):
                text = page.extract_text()
                text_content.append(text)
                print(f"   Extracted page {page_num}/{len(pdf_reader.pages)}")

        full_text = "\n\n".join(text_content)

        # Estimate tokens
        estimated_tokens = len(full_text) / 4
        print(f"✅ Extracted {len(full_text):,} characters (~{int(estimated_tokens):,} tokens)")

        return full_text

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses character-to-token ratio of ~4 chars per token for English prose.
        """
        return len(text) // 4

    def needs_chunking(self, text: str) -> bool:
        """Check if text exceeds the safe context limit and needs chunking."""
        estimated_tokens = self.estimate_tokens(text)
        return estimated_tokens > CONTEXT_TOKEN_LIMIT

    def detect_chapter_markers(self, text: str) -> List[Tuple[int, str]]:
        """Detect chapter and part markers in text.

        Returns list of (position, marker_text) tuples sorted by position.
        Handles various formats including PDF page numbers before markers.
        """
        markers = []

        # Roman numerals pattern (I through XXXIX)
        roman = r'[IVX]+'
        # Written numbers
        written = r'(?:ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|' \
                  r'ELEVEN|TWELVE|THIRTEEN|FOURTEEN|FIFTEEN|SIXTEEN|' \
                  r'SEVENTEEN|EIGHTEEN|NINETEEN|TWENTY|THIRTY|FORTY|FIFTY)'
        # Arabic numbers
        arabic = r'\d{1,3}'

        number = f'(?:{roman}|{written}|{arabic})'

        # Match markers that may be preceded by page numbers (e.g., "6 of 967 PART I")
        # or at line starts
        patterns = [
            # PART markers (with optional page number prefix)
            rf'(?:^|\n|\d+\s+of\s+\d+\s*)(PART\s+{number})\s*(?:\n|$)',
            rf'(?:^|\n|\d+\s+of\s+\d+\s*)(Part\s+{number})\s*(?:\n|$)',
            # CHAPTER markers
            rf'(?:^|\n|\d+\s+of\s+\d+\s*)(CHAPTER\s+{number})\s*(?:\n|$)',
            rf'(?:^|\n|\d+\s+of\s+\d+\s*)(Chapter\s+{number})\s*(?:\n|$)',
            # BOOK markers
            rf'(?:^|\n|\d+\s+of\s+\d+\s*)(BOOK\s+{number})\s*(?:\n|$)',
            rf'(?:^|\n|\d+\s+of\s+\d+\s*)(Book\s+{number})\s*(?:\n|$)',
            # Special sections
            r'(?:^|\n|\d+\s+of\s+\d+\s*)(EPILOGUE)\s*(?:\n|$)',
            r'(?:^|\n|\d+\s+of\s+\d+\s*)(Epilogue)\s*(?:\n|$)',
            r'(?:^|\n|\d+\s+of\s+\d+\s*)(PROLOGUE)\s*(?:\n|$)',
            r'(?:^|\n|\d+\s+of\s+\d+\s*)(Prologue)\s*(?:\n|$)',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE):
                marker_text = match.group(1).strip()
                pos = match.start(1)
                markers.append((pos, marker_text))

        # Remove duplicates and sort by position
        markers = list(set(markers))
        markers.sort(key=lambda x: x[0])

        return markers

    def split_at_markers(self, text: str, markers: List[Tuple[int, str]]) -> List[Dict[str, Any]]:
        """Split text at chapter/part markers."""
        if not markers:
            return [{
                'text': text,
                'start_marker': None,
                'chunk_index': 0,
                'estimated_tokens': self.estimate_tokens(text)
            }]

        chunks = []
        positions = [0] if markers[0][0] > 0 else []
        positions.extend([m[0] for m in markers])
        positions.append(len(text))

        marker_lookup = {m[0]: m[1] for m in markers}

        for i in range(len(positions) - 1):
            start_pos = positions[i]
            end_pos = positions[i + 1]
            chunk_text = text[start_pos:end_pos]

            chunks.append({
                'text': chunk_text,
                'start_marker': marker_lookup.get(start_pos),
                'chunk_index': i,
                'estimated_tokens': self.estimate_tokens(chunk_text)
            })

        # Merge small chunks with previous
        merged = []
        for chunk in chunks:
            if (merged and
                chunk['estimated_tokens'] < MIN_CHUNK_TOKENS and
                merged[-1]['estimated_tokens'] + chunk['estimated_tokens'] < CONTEXT_TOKEN_LIMIT):
                merged[-1]['text'] += chunk['text']
                merged[-1]['estimated_tokens'] += chunk['estimated_tokens']
            else:
                merged.append(chunk)

        for i, chunk in enumerate(merged):
            chunk['chunk_index'] = i

        return merged

    def split_by_tokens(self, text: str, target_tokens: int = None) -> List[Dict[str, Any]]:
        """Split text by token count when no structural markers exist."""
        if target_tokens is None:
            target_tokens = CONTEXT_TOKEN_LIMIT - 10000

        target_chars = target_tokens * 4
        overlap_chars = CHUNK_OVERLAP_TOKENS * 4

        chunks = []
        current_pos = 0
        chunk_index = 0

        while current_pos < len(text):
            end_pos = min(current_pos + target_chars, len(text))

            if end_pos < len(text):
                # Try to find paragraph break near target
                search_start = max(end_pos - 2000, current_pos)
                search_text = text[search_start:end_pos]

                para_break = search_text.rfind('\n\n')
                if para_break > 0:
                    end_pos = search_start + para_break + 2
                else:
                    # Fall back to sentence boundary
                    sentence_breaks = list(re.finditer(r'\.\s', search_text))
                    if sentence_breaks:
                        end_pos = search_start + sentence_breaks[-1].end()
                    else:
                        space_pos = search_text.rfind(' ')
                        if space_pos > 0:
                            end_pos = search_start + space_pos + 1

            chunk_text = text[current_pos:end_pos]

            chunks.append({
                'text': chunk_text,
                'start_marker': f'[Token chunk {chunk_index + 1}]',
                'chunk_index': chunk_index,
                'estimated_tokens': self.estimate_tokens(chunk_text)
            })

            chunk_index += 1

            # If we've reached the end, stop (don't create overlap chunks)
            if end_pos >= len(text):
                break

            # Move forward, applying overlap for context continuity
            current_pos = end_pos - overlap_chars

        return chunks

    def smart_split(self, text: str) -> List[Dict[str, Any]]:
        """Intelligently split text using best available strategy."""
        markers = self.detect_chapter_markers(text)

        if markers:
            print(f"   Found {len(markers)} chapter/part markers")
            chunks = self.split_at_markers(text, markers)

            # Check for oversized chunks
            oversized = [c for c in chunks if c['estimated_tokens'] > CONTEXT_TOKEN_LIMIT]

            if oversized:
                print(f"   Warning: {len(oversized)} chunks exceed limit, sub-splitting...")
                final_chunks = []
                for chunk in chunks:
                    if chunk['estimated_tokens'] > CONTEXT_TOKEN_LIMIT:
                        sub_chunks = self.split_by_tokens(chunk['text'])
                        for j, sub in enumerate(sub_chunks):
                            sub['start_marker'] = f"{chunk['start_marker']} (part {j+1})"
                        final_chunks.extend(sub_chunks)
                    else:
                        final_chunks.append(chunk)

                for i, chunk in enumerate(final_chunks):
                    chunk['chunk_index'] = i

                return final_chunks

            return chunks
        else:
            print(f"   No chapter markers found, using token-based splitting")
            return self.split_by_tokens(text)

    def extract_rooms_from_chunk(self, chunk: Dict[str, Any], chunk_context: str = "") -> List[Dict[str, Any]]:
        """Extract rooms from a single chunk."""
        system_prompt = """Extract all distinct locations from this fiction excerpt and return them as a JSON array.

This is a CHUNK of a larger work. Some locations may have appeared in earlier chunks -
include them if they appear here. Deduplication happens later.

Focus only on locations that appear in THIS excerpt.

For each location provide: name, description (3-5 sentences), exits, items, characters, events, atmosphere.

Output ONLY the JSON array. Start with [ and end with ]."""

        chunk_info = ""
        if chunk.get('start_marker'):
            chunk_info = f"\n\nThis excerpt begins at: {chunk['start_marker']}"

        if chunk_context:
            chunk_info += f"\n\nContext from earlier sections: {chunk_context}"

        user_prompt = f"""Extract all locations from this fiction excerpt as a JSON array.
{chunk_info}

FICTION TEXT:
{chunk['text']}

JSON array:"""

        marker_display = chunk.get('start_marker', 'Start')
        if len(str(marker_display)) > 30:
            marker_display = str(marker_display)[:30] + "..."
        print(f"      {marker_display} (~{chunk['estimated_tokens']:,} tokens)...", end="", flush=True)

        try:
            use_new_param = any(x in self.model.lower() for x in ['gpt-5', 'o1', 'o3', 'o4'])
            token_param = {"max_completion_tokens": MAX_TOKENS} if use_new_param else {"max_tokens": MAX_TOKENS}

            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=TEMPERATURE,
                **token_param,
                stream=True
            )

            response_text = ""
            token_count = 0
            for chunk_resp in stream:
                if chunk_resp.choices[0].delta.content:
                    response_text += chunk_resp.choices[0].delta.content
                    token_count += 1

            print(f" done ({token_count} tokens)")
            response_text = response_text.strip()

            # Parse JSON
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            try:
                rooms = json.loads(response_text)
            except json.JSONDecodeError:
                rooms = self.extract_json_from_response(response_text)

            return rooms if isinstance(rooms, list) else []

        except Exception as e:
            print(f" error: {e}")
            return []

    def process_chunked(self, text: str) -> List[Dict[str, Any]]:
        """Process a large document using chunking strategy."""
        print(f"\n📚 Document exceeds single-pass limit, using chunked processing...")

        chunks = self.smart_split(text)
        print(f"   Split into {len(chunks)} chunks\n")

        all_rooms = []
        room_names_seen = set()

        for i, chunk in enumerate(chunks):
            print(f"📖 Chunk {i + 1}/{len(chunks)}:")

            context = ""
            if room_names_seen:
                recent_rooms = list(room_names_seen)[-20:]
                context = f"Locations seen earlier: {', '.join(recent_rooms)}"

            rooms = self.extract_rooms_from_chunk(chunk, context)

            if rooms:
                print(f"      Found {len(rooms)} rooms")
                all_rooms.extend(rooms)
                for room in rooms:
                    room_names_seen.add(room.get('name', ''))
            else:
                print(f"      Warning: No rooms extracted")

        print(f"\n🔄 Merging {len(all_rooms)} rooms from all chunks...")
        deduped = self.deduplicate_rooms(all_rooms)
        print(f"✅ Final room count after deduplication: {len(deduped)}")

        return deduped

    def deduplicate_rooms(self, rooms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate rooms and merge their information"""
        seen = {}

        for room in rooms:
            name = room.get('name', '').strip()
            if not name:
                continue

            if name not in seen:
                seen[name] = room
            else:
                # Merge information from duplicate
                existing = seen[name]

                # Merge lists
                for key in ['exits', 'items', 'characters', 'events']:
                    if key in room and key in existing:
                        # Combine and deduplicate
                        combined = existing[key] + room[key]
                        existing[key] = list(dict.fromkeys(combined))
                    elif key in room:
                        existing[key] = room[key]

                # Prefer longer description
                if 'description' in room:
                    if len(room['description']) > len(existing.get('description', '')):
                        existing['description'] = room['description']

        return list(seen.values())

    def extract_json_from_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Extract JSON from various response formats with multiple fallback strategies."""
        # Strategy 1: Direct parse (clean JSON response)
        try:
            data = json.loads(response_text)
            if isinstance(data, dict) and "rooms" in data:
                return data["rooms"]
            elif isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract from markdown code blocks
        if "```json" in response_text:
            try:
                json_text = response_text.split("```json")[1].split("```")[0].strip()
                data = json.loads(json_text)
                if isinstance(data, dict) and "rooms" in data:
                    return data["rooms"]
                elif isinstance(data, list):
                    return data
            except (json.JSONDecodeError, IndexError):
                pass

        if "```" in response_text:
            try:
                json_text = response_text.split("```")[1].split("```")[0].strip()
                data = json.loads(json_text)
                if isinstance(data, dict) and "rooms" in data:
                    return data["rooms"]
                elif isinstance(data, list):
                    return data
            except (json.JSONDecodeError, IndexError):
                pass

        # Strategy 3: Find JSON array boundaries
        start = response_text.find('[')
        if start >= 0:
            # Try to find matching close bracket by counting depth
            depth = 0
            for i in range(start, len(response_text)):
                if response_text[i] == '[':
                    depth += 1
                elif response_text[i] == ']':
                    depth -= 1
                    if depth == 0:
                        try:
                            data = json.loads(response_text[start:i+1])
                            if isinstance(data, list):
                                return data
                        except json.JSONDecodeError:
                            pass
                        break

        # Strategy 4: Find JSON object with "rooms" key
        start = response_text.find('{"rooms"')
        if start >= 0:
            depth = 0
            for i in range(start, len(response_text)):
                if response_text[i] == '{':
                    depth += 1
                elif response_text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            data = json.loads(response_text[start:i+1])
                            if isinstance(data, dict) and "rooms" in data:
                                return data["rooms"]
                        except json.JSONDecodeError:
                            pass
                        break

        return []

    def extract_all_rooms(self, text: str) -> List[Dict[str, Any]]:
        """Use LLM with full context to extract all rooms at once"""

        system_prompt = """Extract all distinct locations from this fiction work and return them as a JSON array.

This tool works with multiple formats:
- TV/Film SCRIPTS: Look for [Scene: location] markers, stage directions, and CHARACTER: dialogue
- NOVELS: Extract locations from narrative prose - where characters go, rooms described, settings mentioned
- PLAYS: Look for Act/Scene headers and stage directions

EXAMPLE 1 - TV SCRIPT FORMAT:
[Scene: A grand ceremonial hall]
JOHN: Where are we?
MARY: This is the Great Hall, where the council meets.
[John walks through a doorway into a dusty library]

EXAMPLE 2 - NOVEL FORMAT:
He climbed the stairs to the fourth floor of a grimy tenement building. The room was tiny,
more like a cupboard than proper lodgings. A tattered sofa served as his bed, and a small
table held his few books. Through the thin walls he could hear the neighbors arguing.

EXAMPLE OUTPUT (same JSON format for both):
[
  {
    "name": "Great Hall",
    "description": "A grand ceremonial hall where the council holds important meetings. The space is large and imposing, designed to impress visitors.",
    "exits": ["Library", "Main Entrance"],
    "items": ["council table", "ceremonial banners"],
    "characters": ["John", "Mary"],
    "events": ["Characters arrive and discuss the location"],
    "atmosphere": "formal"
  },
  {
    "name": "Raskolnikov's Garret",
    "description": "A tiny room on the fourth floor of a grimy tenement, more like a cupboard than proper lodgings. A tattered sofa serves as a bed and a small table holds a few books. The thin walls let sounds from neighbors filter through.",
    "exits": ["Stairway", "Hallway"],
    "items": ["tattered sofa", "small table", "books"],
    "characters": ["Raskolnikov"],
    "events": ["Character contemplates his situation"],
    "atmosphere": "cramped"
  }
]

EXTRACTION GUIDELINES:
- Every distinct physical location becomes a separate entry
- Sub-areas within larger spaces are separate rooms (e.g., "Tavern" vs "Tavern - Back Room")
- For novels: infer locations from narrative descriptions (no [Scene:] markers exist)
- Dream sequences, flashbacks, or fantasy locations each get their own entries
- Streets, outdoor areas, and transitional spaces are valid locations
- When a building has multiple significant rooms, create separate entries for each

For each location provide: name, description (3-5 sentences), exits, items, characters, events, atmosphere.

Output ONLY the JSON array. Start with [ and end with ]."""

        user_prompt = f"""Extract all locations from this fiction work as a JSON array.

IMPORTANT GUIDELINES:
- Each distinct physical location is a separate entry
- Sub-areas within buildings get their own entries (e.g., "Tavern" and "Tavern - Private Room" are separate)
- For TV/film scripts: Look for [Scene: X] markers and stage directions
- For novels: Infer locations from narrative prose - where characters go, what rooms are described
- For dreams, visions, or alternate realities: Create separate entries with clear naming
- Streets, bridges, outdoor spaces, and transitional areas are valid locations
- Give rooms descriptive names with context (e.g., "Raskolnikov's Garret" not just "Room")

FICTION TEXT:
{text}

JSON array:"""

        print(f"🤖 Processing entire transcript with full context...")
        print(f"   Using streaming mode for large model...\n")

        try:
            # Use streaming to avoid timeout with slow models
            # Newer models (gpt-5.x, o1, o3, o4) use max_completion_tokens instead of max_tokens
            use_new_param = any(x in self.model.lower() for x in ['gpt-5', 'o1', 'o3', 'o4'])
            token_param = {"max_completion_tokens": MAX_TOKENS} if use_new_param else {"max_tokens": MAX_TOKENS}

            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=TEMPERATURE,
                **token_param,
                stream=True
            )

            # Collect streamed response
            response_text = ""
            token_count = 0
            print("   Generating: ", end="", flush=True)
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    response_text += content
                    token_count += 1
                    # Show progress every 100 tokens
                    if token_count % 100 == 0:
                        print(".", end="", flush=True)

            print(f"\n   Generated ~{token_count} tokens\n")
            response_text = response_text.strip()

            # Debug mode: save response for troubleshooting
            if os.getenv('DEBUG_MODE', 'false').lower() == 'true':
                debug_dir = Path('debug_responses')
                debug_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                debug_file = debug_dir / f'response_{timestamp}.txt'
                debug_file.write_text(
                    f"SYSTEM PROMPT:\n{system_prompt}\n\n"
                    f"USER PROMPT (first 1000 chars):\n{user_prompt[:1000]}...\n\n"
                    f"RESPONSE:\n{response_text}"
                )
                print(f"🐛 Debug response saved to {debug_file}")

            # Extract JSON from response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            rooms = json.loads(response_text)
            print(f"✅ Extracted {len(rooms)} rooms")

            # Deduplicate rooms by name
            deduped = self.deduplicate_rooms(rooms)
            if len(deduped) < len(rooms):
                print(f"🔄 Removed {len(rooms) - len(deduped)} duplicate entries")

            return deduped

        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parsing error: {e}")
            print(f"\n💡 Attempting robust extraction...")

            rooms = self.extract_json_from_response(response_text)
            if rooms:
                print(f"✅ Recovered {len(rooms)} rooms using fallback parser")
                deduped = self.deduplicate_rooms(rooms)
                if len(deduped) < len(rooms):
                    print(f"🔄 Removed {len(rooms) - len(deduped)} duplicate entries")
                return deduped

            # If all strategies failed, show the response for debugging
            print(f"\n📝 Could not parse response. First 500 chars:")
            print("="*60)
            print(response_text[:500])
            print("="*60)
            return []

        except Exception as e:
            print(f"❌ Error processing transcript: {e}")
            import traceback
            traceback.print_exc()
            return []

    def enhance_with_metadata(self, rooms: List[Dict[str, Any]], title: str) -> Dict[str, Any]:
        """Add metadata to the room collection"""

        return {
            "title": title,
            "format": "interactive_fiction_v1",
            "source": "fiction",
            "generated_by": "full_context_processor",
            "processing_method": "single_pass_full_context",
            "room_count": len(rooms),
            "rooms": rooms,
            "metadata": {
                "all_characters": sorted(list(set(
                    char
                    for room in rooms
                    for char in room.get('characters', [])
                ))),
                "all_locations": [room['name'] for room in rooms]
            }
        }

    def process_pdf(self, pdf_path: str, output_path: str = None) -> Dict[str, Any]:
        """Main processing pipeline - handles both small and large documents."""

        print(f"\n{'='*60}")
        print(f"🎬 Processing: {pdf_path}")
        print(f"{'='*60}\n")

        # Extract text
        text = self.extract_text_from_pdf(pdf_path)

        # Check if chunking is needed
        if self.needs_chunking(text):
            estimated = self.estimate_tokens(text)
            print(f"\n⚠️  Document size (~{estimated:,} tokens) exceeds limit ({CONTEXT_TOKEN_LIMIT:,})")
            rooms = self.process_chunked(text)
            processing_method = "chunked"
        else:
            print(f"\n✅ Document fits in context window, using single-pass processing")
            rooms = self.extract_all_rooms(text)
            processing_method = "single_pass"

        if not rooms:
            print(f"\n⚠️  No rooms were extracted!")
            print(f"   The LLM response may have been malformed.")
            print(f"   Try increasing MAX_TOKENS in .env or check the model output.")
            return None

        # Add metadata
        title = Path(pdf_path).stem
        result = self.enhance_with_metadata(rooms, title)
        result['processing_method'] = processing_method

        # Save output
        if output_path is None:
            output_path = Path(pdf_path).stem + "_rooms_full.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print(f"✅ SUCCESS!")
        print(f"{'='*60}")
        print(f"📊 Total unique rooms: {len(rooms)}")
        print(f"💾 Saved to: {output_path}")
        if processing_method == "chunked":
            print(f"📚 Processed with CHUNKING (large document)")
        else:
            print(f"🔥 Processed with FULL CONTEXT (single pass)")
        print(f"{'='*60}\n")

        return result


def get_llm_config(use_remote: bool) -> tuple:
    """Get LLM configuration based on remote/local flag.

    Returns:
        tuple: (api_base, api_key, model, mode_name)
    """
    if use_remote:
        api_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL", "gpt-4o")

        if not api_key or api_key == "sk-your-key-here":
            print("❌ Error: OPENAI_API_KEY not set in .env file")
            print("   Get your API key from: https://platform.openai.com/api-keys")
            sys.exit(1)

        return api_base, api_key, model, "remote (OpenAI)"
    else:
        api_base = os.getenv("LLM_BASE_URL") or os.getenv("LLM_API_BASE", "http://localhost:8000/v1")
        api_key = os.getenv("LLM_API_KEY", "lm-studio")
        model = os.getenv("LLM_MODEL", "local-model")
        return api_base, api_key, model, "local"


def main():
    """Main entry point"""

    parser = argparse.ArgumentParser(
        description="Convert fiction transcripts to interactive fiction game data using LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_transcript_full.py transcript.pdf           # Use local LLM (default)
  python process_transcript_full.py --remote transcript.pdf  # Use OpenAI API
  python process_transcript_full.py -r transcript.pdf out.json

Environment variables:
  Local LLM:  LLM_BASE_URL, LLM_MODEL, LLM_API_KEY
  OpenAI:     OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL
  Shared:     MAX_TOKENS, TEMPERATURE
"""
    )

    # LLM selection (mutually exclusive)
    llm_group = parser.add_mutually_exclusive_group()
    llm_group.add_argument(
        "-r", "--remote",
        action="store_true",
        help="Use OpenAI API (requires OPENAI_API_KEY in .env)"
    )
    llm_group.add_argument(
        "-l", "--local",
        action="store_true",
        help="Use local LLM (default)"
    )

    # Positional arguments
    parser.add_argument(
        "pdf_file",
        help="Path to the PDF transcript file"
    )
    parser.add_argument(
        "output",
        nargs="?",
        help="Output JSON file path (default: <input>_rooms_full.json)"
    )

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.pdf_file):
        print(f"❌ Error: File not found: {args.pdf_file}")
        sys.exit(1)

    # Get LLM configuration
    api_base, api_key, model, mode_name = get_llm_config(args.remote)

    print(f"\n🔧 Using {mode_name} LLM")
    print(f"   API Base: {api_base}")
    print(f"   Model: {model}")
    print(f"   Max Tokens: {MAX_TOKENS}")
    print(f"   Temperature: {TEMPERATURE}")

    # Create processor
    processor = FullContextProcessor(
        api_base=api_base,
        api_key=api_key,
        model=model
    )

    # Process the PDF
    try:
        result = processor.process_pdf(args.pdf_file, args.output)

        if result and result.get('rooms'):
            # Print sample
            print("\n📋 Sample room:")
            print(json.dumps(result['rooms'][0], indent=2))
        else:
            print("\n❌ Processing completed but no valid output was generated")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

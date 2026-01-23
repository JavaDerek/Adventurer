#!/usr/bin/env python3
"""
Doctor Who Transcript to Interactive Fiction Processor (Full Context Version)
Uses the full 128K context window to process entire transcripts at once
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
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


class FullContextProcessor:
    """Process entire Doctor Who transcripts in one pass with full LLM context"""

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

        system_prompt = """Extract all distinct locations from a dramatic script and return them as a JSON array.

EXAMPLE INPUT:
[Scene: The TARDIS materializes in the Panopticon]
DOCTOR: Where are we?
RUNCIBLE: This is the Panopticon, the great hall of Time Lords.
[The Doctor runs through the Matrix, finding himself in a quarry]

EXAMPLE OUTPUT:
[
  {
    "name": "Panopticon",
    "description": "The great ceremonial hall where Time Lords gather for important ceremonies and political events.",
    "exits": ["Panopticon Gallery", "Chancellery"],
    "items": ["ceremonial dais", "viewing galleries"],
    "characters": ["Doctor", "Runcible"],
    "events": ["TARDIS materializes", "ceremony begins"],
    "atmosphere": "formal"
  },
  {
    "name": "Matrix - Quarry",
    "description": "A virtual quarry environment within the Matrix, with steep chalk cliffs and loose rocks.",
    "exits": ["Matrix - Valley"],
    "items": ["chalk cliffs", "loose rocks"],
    "characters": ["Doctor"],
    "events": ["Doctor is pursued through quarry"],
    "atmosphere": "dangerous"
  }
]

Extract EVERY location from the script. Each distinct area is a separate entry:
- Physical locations (TARDIS, Panopticon, Chancellery, Museum, etc.)
- Virtual/Matrix locations - each distinct setting is its own room (Quarry, Battlefield, Pool, Marsh, etc.)
- Sub-areas within larger spaces (Panopticon vs Panopticon Gallery vs Panopticon Vault)

For each location provide: name, description (3-5 sentences), exits, items, characters, events, atmosphere.

Output ONLY the JSON array. Start with [ and end with ]."""

        user_prompt = f"""Extract all locations from this script as a JSON array.

Remember:
- Each physical location is a separate entry (TARDIS, Panopticon, Chancellery, Records Room, Museum, etc.)
- Panopticon, Panopticon Gallery, and Panopticon Vault are THREE separate rooms
- Each Matrix environment is a separate entry (Matrix - Quarry, Matrix - Battlefield, Matrix - Pool, etc.)
- Look for the actual setting in Matrix scenes (quarry, marsh, railway, etc.) and create separate rooms for each

SCRIPT:
{text}

JSON array:"""

        print(f"🤖 Processing entire transcript with full context...")
        print(f"   Using streaming mode for large model...\n")

        try:
            # Use streaming to avoid timeout with slow models
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
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
            "source": "doctor_who_transcript",
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
        """Main processing pipeline"""

        print(f"\n{'='*60}")
        print(f"🎬 Processing (Full Context): {pdf_path}")
        print(f"{'='*60}\n")

        # Extract text
        text = self.extract_text_from_pdf(pdf_path)

        # Process entire document at once
        rooms = self.extract_all_rooms(text)

        if not rooms:
            print(f"\n⚠️  No rooms were extracted!")
            print(f"   The LLM response may have been malformed.")
            print(f"   Try increasing MAX_TOKENS in .env or check the model output.")
            return None

        # Add metadata
        title = Path(pdf_path).stem
        result = self.enhance_with_metadata(rooms, title)

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
        print(f"🔥 Processed with FULL CONTEXT (no chunking!)")
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

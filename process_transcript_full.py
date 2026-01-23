#!/usr/bin/env python3
"""
Doctor Who Transcript to Interactive Fiction Processor (Full Context Version)
Uses the full 128K context window to process entire transcripts at once
"""

import json
import os
import sys
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

    def extract_all_rooms(self, text: str) -> List[Dict[str, Any]]:
        """Use LLM with full context to extract all rooms at once"""

        system_prompt = """You are a JSON data extraction system. You analyze dramatic scripts and output structured JSON data.

Example output format:
[
  {
    "name": "TARDIS Console Room",
    "description": "The central control room of the TARDIS with a hexagonal console.",
    "exits": ["TARDIS Corridor"],
    "items": ["Time Rotor", "Console"],
    "characters": ["The Doctor"],
    "events": ["Doctor receives distress signal"],
    "atmosphere": "mysterious"
  }
]

Your task is to analyze the ENTIRE script and identify EVERY distinct LOCATION/ROOM that appears in the story.

For each unique location, extract:
1. **name**: Clear, concise location name (e.g., "TARDIS Console Room", "Panopticon", "Matrix - Quarry")
2. **description**: Vivid description combining ALL mentions of this location throughout the script (3-5 sentences)
3. **exits**: Array of named connections to other rooms mentioned in the script
4. **items**: All notable objects, props, or features mentioned across all scenes in this location
5. **characters**: All characters who appear in this location throughout the story
6. **events**: Key plot events that occur here (chronological order if possible)
7. **atmosphere**: Overall mood/tone (e.g., "tense", "mysterious", "formal")

CRITICAL RULES:
- Include ALL locations: real, virtual, dream sequences, Matrix simulations, visions, etc.
- Each distinct playable area is a separate room (e.g., "Cliff Face", "Valley", "Pool" are different from "Matrix")
- If a location appears multiple times with ONLY capitalization differences (e.g., "Tardis" vs "TARDIS"), consolidate them
- BUT locations that are actually DIFFERENT (e.g., "Panopticon" vs "Panopticon Gallery") should be SEPARATE entries
- Virtual/dream/Matrix locations are REAL game locations - include them all!
- Sub-locations within larger areas (like different parts of the Matrix) should be separate rooms

EXAMPLES:
- "TARDIS Console Room" and "Tardis console room" → ONE entry
- "Matrix - Quarry" and "Matrix - Battlefield" → TWO entries (different locations in the Matrix)
- "Panopticon" and "Panopticon Gallery" → TWO entries (different rooms)

OUTPUT FORMAT - CRITICAL:
You MUST output ONLY a raw JSON array. Your entire response must be valid JSON that starts with [ and ends with ].
DO NOT write any markdown, explanations, comments, or code blocks.
DO NOT use ```json or ``` markers.
DO NOT write "Here is the JSON" or any other text.
Your response must be ONLY the JSON array, nothing else."""

        user_prompt = f"""Analyze this complete Doctor Who script and extract ALL unique locations/rooms as a JSON array.

IMPORTANT: Include EVERY location in the script as SEPARATE entries:

Physical locations:
- TARDIS Console Room
- Panopticon
- Panopticon Gallery (separate from Panopticon!)
- Panopticon Vault (separate from Panopticon!)
- Chancellery
- Records Room
- Museum
- etc.

Matrix/Virtual locations (these are SEPARATE rooms even though they're all "in the Matrix"):
- Matrix - Operating Theatre
- Matrix - Betchworth Quarry
- Matrix - World War One Battlefield
- Matrix - Railway Junction
- Matrix - Cliff Face
- Matrix - Valley
- Matrix - Reed Thicket
- Matrix - Pool
- Matrix - Marsh
- etc.

If the scene description says [Matrix] or similar, look at the ACTUAL setting described (quarry, battlefield, pool, etc.) and create a separate room for each one.

Only consolidate entries that are literally the same place with different capitalization (e.g., "Tardis" vs "TARDIS").

---
{text}
---

Output the JSON array now. Start your response with [ and end with ]. Do not write anything except the JSON array:"""

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
            print(f"\n📝 FULL RESPONSE FROM MODEL:")
            print("="*60)
            print(response_text)
            print("="*60)

            # Try to salvage partial JSON
            print(f"\n💡 Attempting to extract partial results...")
            try:
                # Find the start of the array
                start = response_text.find('[')
                if start >= 0:
                    # Try to find matching close bracket
                    depth = 0
                    for i in range(start, len(response_text)):
                        if response_text[i] == '[':
                            depth += 1
                        elif response_text[i] == ']':
                            depth -= 1
                            if depth == 0:
                                partial = response_text[start:i+1]
                                rooms = json.loads(partial)
                                print(f"✅ Recovered {len(rooms)} rooms from partial JSON")
                                return rooms
            except:
                pass

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


def main():
    """Main entry point"""

    if len(sys.argv) < 2:
        print("Usage: python process_transcript_full.py <pdf_file> [output.json]")
        print("\nThis version uses FULL CONTEXT (no chunking) for better results!")
        print("Requires a model with large context window (32K+ recommended)")
        print("\nEnvironment variables:")
        print(f"  LLM_BASE_URL: {LLM_API_BASE}")
        print(f"  LLM_MODEL: {LLM_MODEL}")
        print(f"  MAX_TOKENS: {MAX_TOKENS}")
        print(f"  TEMPERATURE: {TEMPERATURE}")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(pdf_path):
        print(f"❌ Error: File not found: {pdf_path}")
        sys.exit(1)

    # Create processor
    processor = FullContextProcessor(
        api_base=LLM_API_BASE,
        api_key=LLM_API_KEY,
        model=LLM_MODEL
    )

    # Process the PDF
    try:
        result = processor.process_pdf(pdf_path, output_path)

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

#!/usr/bin/env python3
"""
Test your LLM connection (local or remote)
"""

import argparse
import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


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


def test_connection(use_remote: bool = False):
    """Test connection to LLM"""

    api_base, api_key, model, mode_name = get_llm_config(use_remote)

    print(f"🔍 Testing connection to {mode_name} LLM...")
    print(f"   API Base: {api_base}")
    print(f"   Model: {model}\n")

    try:
        client = OpenAI(base_url=api_base, api_key=api_key)

        # Test 1: List models
        print("📋 Step 1: Listing available models...")
        try:
            models = client.models.list()
            print(f"✅ Success! Found {len(models.data)} model(s):")
            for model_info in models.data:
                print(f"   - {model_info.id}")
        except Exception as e:
            print(f"⚠️  Could not list models: {e}")
            print("   (This is okay if your server doesn't support model listing)")

        # Test 2: Simple completion
        print(f"\n💬 Step 2: Testing chat completion...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello, I am working!' and nothing else."}
            ],
            max_tokens=50,
            temperature=0.7
        )

        reply = response.choices[0].message.content
        print(f"✅ Success! Model responded:")
        print(f"   '{reply}'\n")

        # Test 3: JSON extraction (what we'll use in the main script)
        print(f"💻 Step 3: Testing JSON extraction...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a JSON generator. Always respond with valid JSON."},
                {"role": "user", "content": 'Create a JSON object with one key "status" and value "ready". Return only the JSON, nothing else.'}
            ],
            max_tokens=100,
            temperature=0.3
        )

        reply = response.choices[0].message.content.strip()
        print(f"✅ Success! Model responded:")
        print(f"   {reply}\n")

        # Try to parse as JSON
        import json
        if "```json" in reply:
            reply = reply.split("```json")[1].split("```")[0].strip()
        elif "```" in reply:
            reply = reply.split("```")[1].split("```")[0].strip()

        try:
            parsed = json.loads(reply)
            print(f"✅ JSON is valid: {parsed}\n")
        except json.JSONDecodeError:
            print(f"⚠️  Response is not valid JSON")
            print(f"   You may need to adjust your model or prompts\n")

        print("="*60)
        print("🎉 CONNECTION TEST COMPLETE!")
        print("="*60)
        print("Your local LLM is ready to use with process_transcript.py")
        print("\nNext steps:")
        print("  1. Run: python process_transcript.py 'your-transcript.pdf'")
        print("  2. Check the generated JSON file")
        print("="*60)

        return True

    except ConnectionError as e:
        print(f"\n❌ Connection failed: {e}")
        print(f"\n🔧 Troubleshooting:")
        print(f"   1. Make sure your local LLM server is running")
        print(f"   2. Check the API endpoint: {api_base}")
        print(f"   3. Try: curl {api_base}/models")
        return False

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test LLM connection (local or remote)"
    )

    llm_group = parser.add_mutually_exclusive_group()
    llm_group.add_argument(
        "-r", "--remote",
        action="store_true",
        help="Test OpenAI API connection"
    )
    llm_group.add_argument(
        "-l", "--local",
        action="store_true",
        help="Test local LLM connection (default)"
    )

    args = parser.parse_args()

    success = test_connection(use_remote=args.remote)
    sys.exit(0 if success else 1)

"""
Gemini-powered Sanskrit translation CLI.

Usage:
    python translate_gemini.py --text "<SANSKRIT>"
    python translate_gemini.py --file path/to/sanskrit.txt

Env:
    GEMINI_API_KEY must be set (or in .env)
"""
import os
import sys
import argparse
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("ERROR: GEMINI_API_KEY not set. Add to .env or export environment.")
    sys.exit(1)

import google.generativeai as genai

genai.configure(api_key=API_KEY)

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

PROMPT_TEMPLATE = (
    "You are an expert Sanskrit-to-Hindi and Sanskrit-to-English translator.\n"
    "Translate the following Sanskrit text into both Hindi and English.\n"
    "Preserve poetic meaning and avoid literal word-by-word translation.\n"
    "If any verse is incomplete, intelligently reconstruct and translate meaningfully.\n\n"
    "Sanskrit Text:\n---------------------------------------\n"
    "{text}\n"
    "---------------------------------------\n\n"
    "Output format:\n"
    "1. Corrected Sanskrit (if needed)\n"
    "2. Hindi Meaning:\n"
    "3. English Meaning:\n"
)


def translate_sanskrit(text: str) -> str:
    if not text or not text.strip():
        raise ValueError("Empty Sanskrit text provided")
    prompt = PROMPT_TEMPLATE.format(text=text.strip())
    model = genai.GenerativeModel(MODEL_NAME)
    resp = model.generate_content(prompt)
    # Support streaming and non-streaming API differences
    if hasattr(resp, 'text') and resp.text:
        return resp.text.strip()
    # Fallback: concatenate parts
    if hasattr(resp, 'candidates') and resp.candidates:
        parts = []
        for c in resp.candidates:
            if hasattr(c, 'content') and hasattr(c.content, 'parts'):
                for p in c.content.parts:
                    if hasattr(p, 'text'):
                        parts.append(p.text)
        if parts:
            return "\n".join(parts).strip()
    raise RuntimeError("Gemini API returned no text")


def main(argv: Optional[list] = None):
    parser = argparse.ArgumentParser(description="Translate Sanskrit to Hindi & English via Gemini")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Direct Sanskrit text input")
    group.add_argument("--file", type=str, help="Path to a file containing Sanskrit text")
    args = parser.parse_args(argv)

    if args.file:
        if not os.path.exists(args.file):
            print(f"ERROR: File not found: {args.file}")
            sys.exit(1)
        with open(args.file, 'r', encoding='utf-8') as f:
            input_text = f.read()
    else:
        input_text = args.text

    try:
        output = translate_sanskrit(input_text)
        print(output)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()


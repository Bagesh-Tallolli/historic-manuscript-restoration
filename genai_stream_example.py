"""
Example: Using google-genai Client to stream content from Gemini.
Dependencies: pip install google-genai
Env: export GEMINI_API_KEY=your_key
"""
import os
from google import genai
from google.genai import types

def pick_model(client) -> str:
    try:
        models = list(client.models.list())
        names = [m.name for m in models]
        # Filter out embeddings or non-generation models
        gen_names = [n for n in names if "gemini" in n and not any(x in n for x in ["embedding", "gecko", "vision", "image"])]
        # Prefer latest flash/pro variants
        preferred_order = [
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-2.0-flash-exp",
            "gemini-1.0-pro",
            "gemini-1.0-pro-latest",
            "gemini-1.0-pro-001",
            "gemini-1.0-pro-002",
        ]
        for p in preferred_order:
            if p in gen_names:
                return p
        if gen_names:
            return gen_names[0]
    except Exception:
        pass
    return "gemini-1.5-flash"

def generate(input_text: str):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment")

    client = genai.Client(api_key=api_key)

    model = pick_model(client)
    print(f"Using model: {model}")

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=input_text)],
        )
    ]

    # Optional tools (Google Search)
    tools = [types.Tool(googleSearch=types.GoogleSearch())]

    # Thinking config is set via dedicated type
    thinking_cfg = types.ThinkingConfig(thinkingLevel=types.ThinkingLevel.HIGH)

    generate_cfg = types.GenerateContentConfig(
        tools=tools,
        thinkingConfig=thinking_cfg,
    )

    # Stream response
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_cfg,
    ):
        if hasattr(chunk, "text") and chunk.text:
            print(chunk.text, end="")

if __name__ == "__main__":
    # Replace with your desired input
    generate("INSERT_INPUT_HERE")

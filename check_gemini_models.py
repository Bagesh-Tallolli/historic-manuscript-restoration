"""
Test script to check available Gemini models and their capabilities
"""
from google import genai

from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY", "")

print("=" * 70)
print("Checking Available Gemini Models")
print("=" * 70)

try:
    client = genai.Client(api_key=API_KEY)
    models = list(client.models.list())

    print(f"\nFound {len(models)} models\n")

    generation_models = []

    for model in models:
        print(f"\n📦 Model: {model.name}")

        # Check supported methods
        if hasattr(model, 'supported_generation_methods'):
            methods = model.supported_generation_methods or []
            print(f"   Supported methods: {', '.join(methods) if methods else 'None listed'}")

            if 'generateContent' in methods:
                generation_models.append(model.name)
                print(f"   ✅ Supports generateContent")
            else:
                print(f"   ❌ Does NOT support generateContent")
        else:
            print(f"   ⚠️  No supported_generation_methods attribute")
            if 'gemini' in model.name.lower():
                generation_models.append(model.name)
                print(f"   ✅ Assuming supports generateContent (Gemini model)")

        # Check other attributes
        if hasattr(model, 'display_name'):
            print(f"   Display name: {model.display_name}")
        if hasattr(model, 'description'):
            desc = model.description[:100] + "..." if len(model.description) > 100 else model.description
            print(f"   Description: {desc}")

    print("\n" + "=" * 70)
    print("SUMMARY: Models that support generateContent")
    print("=" * 70)

    if generation_models:
        for i, model_name in enumerate(generation_models, 1):
            print(f"{i}. {model_name}")
    else:
        print("⚠️  No models found that explicitly support generateContent")

    print("\n" + "=" * 70)
    print("RECOMMENDED CONFIGURATION")
    print("=" * 70)

    if generation_models:
        print("\nUpdate PREFERRED_MODELS in gemini_ocr_streamlit_v2.py to:")
        print("PREFERRED_MODELS = [")
        for model_name in generation_models[:4]:  # Top 4
            # Extract just the model name without 'models/' prefix
            short_name = model_name.replace('models/', '')
            print(f'    "{short_name}",')
        print("]")

        print(f"\n✅ Recommended default: {generation_models[0]}")
    else:
        print("\n⚠️  Could not determine available models")
        print("Try using 'models/gemini-1.5-flash' as fallback")

except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nPossible issues:")
    print("1. API key invalid or expired")
    print("2. Network connectivity")
    print("3. API version mismatch")

print("\n" + "=" * 70)


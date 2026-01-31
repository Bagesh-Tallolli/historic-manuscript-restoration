"""
Quick test of model selection fix
"""
from google import genai

API_KEY = "AIzaSyAxTN1rfdyJQVE3ecy5d8Zqkl5I431nBh0"

print("Testing model selection...")

try:
    client = genai.Client(api_key=API_KEY)

    # Test with the models we're using in the app
    test_models = [
        "gemini-2.5-flash",
        "models/gemini-2.5-flash",
        "gemini-2.0-flash",
        "models/gemini-2.0-flash",
    ]

    for model_name in test_models:
        try:
            print(f"\n✓ Testing: {model_name}")

            # Try to use the model (simple text generation)
            response = client.models.generate_content(
                model=model_name,
                contents="Say hello"
            )

            if response and response.text:
                print(f"  ✅ SUCCESS! Model works")
                print(f"  Response: {response.text[:50]}...")
            else:
                print(f"  ⚠️  No response text")

        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg or "NOT_FOUND" in error_msg:
                print(f"  ❌ Model not found: {model_name}")
            else:
                print(f"  ❌ Error: {error_msg[:100]}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    # Test the working format
    print("\nTrying recommended format: models/gemini-2.5-flash")
    try:
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents="Test"
        )
        print("✅ Recommended format WORKS!")
        print(f"   Use: 'models/gemini-2.5-flash' (with 'models/' prefix)")
    except Exception as e:
        print(f"❌ Even recommended format failed: {e}")

except Exception as e:
    print(f"❌ Client initialization failed: {e}")


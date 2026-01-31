"""
Streamlit app: Upload an image and translate Sanskrit text using Gemini via google-genai client.
- Hardcoded API key provided by user
- Sends the image along with the specified system prompt
- Displays structured output
"""
import os
import io
import streamlit as st
from PIL import Image

# google-genai client
from google import genai
from google.genai import types
# Add Groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False

# Load API keys from environment variables
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY", "")
API_KEY_SOURCE = "ENVIRONMENT" if API_KEY else "NOT_FOUND"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Default preferred models order
PREFERRED_MODELS = [
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash-exp",
]

def _model_name_to_full(name: str, available: list[str]) -> str:
    # Map simple name like 'gemini-1.5-pro' to full 'models/gemini-1.5-pro' if present
    if name in available:
        return name
    prefixed = f"models/{name}"
    return prefixed if prefixed in available else name

# SYSTEM_PROMPT = (
#     "You are an expert Sanskrit-to-Hindi and Sanskrit-to-English translator.\n"
#     "Translate the following Sanskrit text into both Hindi and English.\n"
#     "Preserve poetic meaning and avoid literal word-by-word translation.\n"
#     "If any verse is incomplete, intelligently reconstruct and translate meaningfully.\n\n"
#     "Sanskrit Text:\n---------------------------------------\n"
#     "<PASTE YOUR SANSKRIT TEXT HERE>\n"
#     "---------------------------------------\n\n"
#     "Output format:\n"
#     "1. Corrected Sanskrit (if needed)\n"
#     "2. Hindi Meaning:\n"
#     "3. English Meaning:\n"
# )

SYSTEM_PROMPT = (
    "You are an expert Sanskrit-to-Hindi and Sanskrit-to-English translator.\n"
    "Translate the following Sanskrit text into both Hindi and English.\n"
    "Preserve poetic meaning and avoid literal word-by-word translation.\n"
    "If any verse is incomplete, intelligently reconstruct and translate meaningfully.\n\n"
    "Sanskrit Text:\n---------------------------------------\n"
    "<PASTE YOUR SANSKRIT TEXT HERE>\n"
    "---------------------------------------\n\n"
    "Output format:\n"
    "1. Corrected Sanskrit (if needed)\n"
    "2. Hindi Meaning:\n"
    "3. English Meaning:\n"
)

# UI config
st.set_page_config(page_title="Gemini Sanskrit OCR & Translation", page_icon="📜", layout="centered")
st.title("📜 Gemini Sanskrit OCR & Translation")
st.caption("Upload an image with Sanskrit text and get Hindi & English translations.")

# Safety: show API status
if not API_KEY:
    st.error("Gemini API key not found. The application uses a hardcoded key.")
    st.stop()
else:
    st.info(f"Using API key from: {API_KEY_SOURCE}")

# Helper: choose a generation-capable model
def pick_model(client: genai.Client, preferred=PREFERRED_MODELS) -> str:
    try:
        models = list(client.models.list())
        names = [m.name for m in models]
        # Filter generation-capable models
        gen_names = [n for n in names if "gemini" in n and not any(x in n for x in ["embedding", "gecko", "vision", "image"]) ]
        # Try preferred order mapped to full names
        for p in preferred:
            full = _model_name_to_full(p, gen_names)
            if full in gen_names:
                return full
        if gen_names:
            return gen_names[0]
    except Exception:
        pass
    return "models/gemini-1.5-flash"

# File uploader
uploaded = st.file_uploader("Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"])

with st.expander("Advanced options", expanded=False):
    custom_prompt = st.text_area("Override system prompt (optional)", value=SYSTEM_PROMPT, height=200)
    selected_model_friendly = st.selectbox("Preferred model (Gemini)", options=PREFERRED_MODELS + ["auto"], index=0, help="Pick a Gemini model or 'auto'.")
    provider = st.selectbox("Provider", options=["Gemini (google-genai)", "Groq (text-only)"])
    groq_model = st.text_input("Groq model", value="openai/gpt-oss-20b", help="Text-only model; images aren’t supported.")

run_btn = st.button("Translate with selected provider")

# If Groq is selected, ask for text input (since Groq model is text-only)
if 'groq_input' not in st.session_state:
    st.session_state.groq_input = ""
if provider == "Groq (text-only)":
    st.info("Groq chat model is text-only. Please paste the Sanskrit text below.")
    st.session_state.groq_input = st.text_area("Sanskrit text (for Groq)", value=st.session_state.groq_input, height=150)

# Output containers
result_placeholder = st.empty()

if run_btn:
    if provider == "Groq (text-only)":
        if not GROQ_AVAILABLE:
            st.error("Groq library not installed. Install with: pip install groq")
            st.stop()
        if not st.session_state.groq_input.strip():
            st.warning("Please paste Sanskrit text for Groq.")
            st.stop()
        # Build Groq messages with system prompt + user text
        user_text = st.session_state.groq_input.strip()
        prompt_text = (custom_prompt or SYSTEM_PROMPT).replace("<PASTE YOUR SANSKRIT TEXT HERE>", user_text)
        try:
            client = Groq(api_key=GROQ_API_KEY)
            completion = client.chat.completions.create(
                model=groq_model,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=1,
                max_completion_tokens=2048,
                top_p=1,
                reasoning_effort="medium",
                stream=True,
                stop=None,
            )
            streamed = []
            for chunk in completion:
                delta = getattr(chunk.choices[0].delta, 'content', '') or ''
                if delta:
                    streamed.append(delta)
            output_text = ''.join(streamed).strip()
            if not output_text:
                st.warning("No text returned by Groq.")
            else:
                st.subheader("Translation Output (Groq)")
                result_placeholder.markdown(output_text)
        except Exception as e:
            st.error(f"Groq request failed: {e}")
        st.stop()

    # Gemini flow continues as before
    if not uploaded:
        st.warning("Please upload an image first.")
        st.stop()

    # Read image
    try:
        img_bytes = uploaded.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        st.error(f"Failed to read image: {e}")
        st.stop()

    # Show image preview (use_container_width)
    st.image(image, caption="Uploaded image", use_container_width=True)

    # Prepare image part for google-genai (use raw bytes)
    img_buf = io.BytesIO()
    image.save(img_buf, format="JPEG", quality=90)
    jpg_bytes = img_buf.getvalue()
    image_part = types.Part.from_bytes(mime_type="image/jpeg", data=jpg_bytes)

    # Initialize client and pick model
    try:
        client = genai.Client(api_key=API_KEY)
        # Build available names for mapping
        available_names = [m.name for m in client.models.list()]
        if selected_model_friendly == "auto":
            model_name = pick_model(client)
        else:
            model_name = _model_name_to_full(selected_model_friendly, available_names)
        st.caption(f"Model selected: {model_name}")
    except Exception as e:
        st.error(f"Gemini client init error: {e}")
        st.stop()

    # Compose contents: prompt + image
    prompt_text = custom_prompt or SYSTEM_PROMPT
    contents = [
        types.Content(role="user", parts=[
            types.Part.from_text(text=prompt_text),
            image_part
        ])
    ]

    # Optional tools: none required here
    thinking_cfg = types.ThinkingConfig(thinkingLevel=types.ThinkingLevel.HIGH)
    generate_cfg = types.GenerateContentConfig(thinkingConfig=thinking_cfg)

    # Call streaming API and accumulate text with one retry on 429
    def call_stream(model_nm: str, img_bytes_local: bytes):
        local_image_part = types.Part.from_bytes(mime_type="image/jpeg", data=img_bytes_local)
        local_contents = [
            types.Content(role="user", parts=[
                types.Part.from_text(text=prompt_text),
                local_image_part
            ])
        ]
        streamed = []
        for chunk in client.models.generate_content_stream(
            model=model_nm,
            contents=local_contents,
            config=generate_cfg,
        ):
            if hasattr(chunk, "text") and chunk.text:
                streamed.append(chunk.text)
        return "".join(streamed).strip()

    try:
        output_text = call_stream(model_name, jpg_bytes)
    except Exception as e:
        msg = str(e)
        if "NOT_FOUND" in msg or "404" in msg:
            # Auto-switch to first compatible model
            st.warning("Selected model not available for generateContent. Switching to first compatible model...")
            try:
                model_name = pick_model(client)
                output_text = call_stream(model_name, jpg_bytes)
                st.caption(f"Switched to: {model_name}")
            except Exception as e3:
                st.error(f"Gemini request failed after model switch: {e3}")
                st.stop()
        elif "RESOURCE_EXHAUSTED" in msg or "429" in msg:
            st.warning("Quota or rate limit hit. Falling back to gemini-1.5-flash and reduced image quality, retrying once...")
            img_buf2 = io.BytesIO()
            image.save(img_buf2, format="JPEG", quality=70)
            jpg_bytes2 = img_buf2.getvalue()
            fallback_model = _model_name_to_full("gemini-1.5-flash", available_names)
            try:
                output_text = call_stream(fallback_model, jpg_bytes2)
                st.caption(f"Retried with model: {fallback_model} and compressed image")
            except Exception as e2:
                st.error(f"Gemini request failed after retry: {e2}")
                st.stop()
        else:
            st.error(f"Gemini request failed: {e}")
            st.stop()

    if not output_text:
        st.warning("No text returned by Gemini.")
    else:
        st.subheader("Translation Output")
        result_placeholder.markdown(output_text)

    st.info("Tip: If the output format isn't structured, adjust the prompt in Advanced options.")

# Footer
st.markdown("---")
st.markdown("Using google-genai client with hardcoded API key.")

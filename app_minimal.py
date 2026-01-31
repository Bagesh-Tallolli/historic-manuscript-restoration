"""Minimal Streamlit App: Kaggle Restoration + Tesseract Sanskrit OCR

Workflow:
1. Upload image
2. Restore with selected Kaggle ViT checkpoint (optional skip)
3. Run multi-pass Tesseract OCR (lang san fallback hin/dev/eng)
4. Display Devanagari text + basic stats

Requires:
- Tesseract installed with Sanskrit traineddata (tesseract-ocr-san)
- Dependencies: streamlit, torch, torchvision, pytesseract, opencv-python, Pillow, numpy

Run:
    streamlit run app_minimal.py
"""

import streamlit as st
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from PIL import Image
import pytesseract

from models.vit_restorer import create_vit_restorer
from utils.image_restoration_enhanced import create_enhanced_restorer
from ocr.preprocess import OCRPreprocessor

# ------------------ Configuration ------------------
st.set_page_config(page_title="Sanskrit Restoration & OCR (Minimal)", page_icon="🕉️", layout="wide")
st.title("🕉️ Sanskrit Manuscript Restoration & OCR (Minimal)")
st.caption("Kaggle ViT restoration → Multi-pass Tesseract (Sanskrit)")

# Sidebar: Model selection
st.sidebar.header("🧠 Restoration Model")
MODEL_DIR = Path("checkpoints/kaggle")
AVAILABLE_MODELS = {
    "final.pth": MODEL_DIR / "final.pth",
    "desti.pth": MODEL_DIR / "desti.pth",
    "final_converted.pth": MODEL_DIR / "final_converted.pth",
    "Skip Restoration": None,
}
# Verification helpers
def verify_restoration_model(model):
    if model is None:
        return {"status": "skipped"}
    try:
        dummy = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            out = model(dummy)
        params = sum(p.numel() for p in model.parameters())
        return {
            "status": "ok" if out.shape == (1, 3, 256, 256) else "shape_mismatch",
            "output_shape": tuple(out.shape),
            "param_count_million": round(params / 1_000_000, 2)
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

model_choice = st.sidebar.selectbox("Select Kaggle Model", list(AVAILABLE_MODELS.keys()), index=0)
model_path = AVAILABLE_MODELS[model_choice]

st.sidebar.header("📝 OCR Settings")
multi_pass = st.sidebar.checkbox("Multi-pass OCR", value=True)
show_preprocess = st.sidebar.checkbox("Show preprocessed image", value=False)

# Session state for model
def load_restoration(model_path: Optional[Path]):
    if model_path is None:
        return None, None
    if not model_path.exists():
        st.warning(f"Model not found: {model_path}")
        return None, None
    # Load checkpoint
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    # Heuristic architecture selection
    has_head = any('head.' in k for k in state_dict)
    has_patch = any('patch_recon' in k for k in state_dict)
    has_skip = any('skip_fusion' in k for k in state_dict)
    if has_head and not has_patch:
        model = create_vit_restorer('base', img_size=256, use_skip_connections=False, use_simple_head=True)
    elif has_patch and has_skip:
        model = create_vit_restorer('base', img_size=256, use_skip_connections=True, use_simple_head=False)
    else:
        model = create_vit_restorer('base', img_size=256)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval()
    enhanced = create_enhanced_restorer(model, device='cpu', patch_size=256, overlap=32)
    return model, enhanced

if 'restoration_model' not in st.session_state:
    st.session_state.restoration_model, st.session_state.enhanced_restorer = load_restoration(model_path)
    st.session_state.restoration_model_info = verify_restoration_model(st.session_state.restoration_model)

# Allow re-load if model choice changes
if st.sidebar.button("🔁 Reload Model"):
    st.session_state.restoration_model, st.session_state.enhanced_restorer = load_restoration(model_path)
    st.session_state.restoration_model_info = verify_restoration_model(st.session_state.restoration_model)
    st.success("Model reloaded & verified.")

# Sidebar status block
with st.sidebar.expander("📊 Model Status", expanded=True):
    info = st.session_state.get('restoration_model_info', {})
    status = info.get('status')
    if status == 'ok':
        st.success(f"Loaded ✓ | Output: {info.get('output_shape')} | Params: {info.get('param_count_million')}M")
    elif status == 'skipped':
        st.info("Restoration skipped.")
    elif status == 'shape_mismatch':
        st.warning(f"Loaded but unexpected output shape: {info.get('output_shape')}")
    elif status == 'error':
        st.error(f"Model error: {info.get('error')}")
    else:
        st.warning("Model not verified yet.")
    if model_path and model_path is not None:
        st.caption(f"Checkpoint: {model_path}")
# Manual re-verify button
if st.sidebar.button("✅ Re-Verify Model"):
    st.session_state.restoration_model_info = verify_restoration_model(st.session_state.restoration_model)
    st.sidebar.success("Verification complete.")

# ------------------ Upload ------------------
st.markdown("### 📤 Upload Image")
uploaded = st.file_uploader("Upload Sanskrit manuscript image", type=["jpg","jpeg","png","tif","tiff","bmp"])

preprocessor = OCRPreprocessor()

def restore_image(img_np: np.ndarray) -> np.ndarray:
    if st.session_state.enhanced_restorer is None:
        return img_np
    h, w = img_np.shape[:2]
    if h > 512 or w > 512:
        restored = st.session_state.enhanced_restorer.restore_image(img_np, use_patches=True, apply_postprocess=True)
    else:
        restored = st.session_state.enhanced_restorer.restore_image(img_np, use_patches=False, apply_postprocess=True)
    if restored.dtype != np.uint8:
        if restored.max() <= 1.0:
            restored = (restored * 255).clip(0,255).astype(np.uint8)
        else:
            restored = restored.clip(0,255).astype(np.uint8)
    return restored

def tesseract_lang():
    try:
        langs = pytesseract.get_languages()
        for candidate in ['san','hin','dev','eng']:
            if candidate in langs:
                return candidate
    except Exception:
        pass
    return 'eng'

TES_LANG = tesseract_lang()

def ocr_multipass(image_np: np.ndarray, lang: str, multi: bool=True):
    pil_img = Image.fromarray(image_np)
    configs = [
        f'--oem 3 --psm 6 -l {lang}',
        f'--oem 3 --psm 4 -l {lang}',
        f'--oem 3 --psm 3 -l {lang}',
        f'--oem 1 --psm 6 -l {lang}',
    ]
    results = []
    confidences = []
    if multi:
        for cfg in configs:
            try:
                txt = pytesseract.image_to_string(pil_img, config=cfg).strip()
                data = pytesseract.image_to_data(pil_img, config=cfg, output_type=pytesseract.Output.DICT)
                conf = [int(c) for c in data['conf'] if c != '-1']
                avg_conf = float(np.mean(conf)/100.0) if conf else 0.0
                if txt:
                    results.append(txt)
                    confidences.append(avg_conf)
            except Exception:
                continue
        if results:
            scores = [len(t)*c for t,c in zip(results, confidences)]
            best = int(np.argmax(scores))
            return results[best], confidences[best]
        return "", 0.0
    else:
        cfg = configs[0]
        try:
            txt = pytesseract.image_to_string(pil_img, config=cfg).strip()
            data = pytesseract.image_to_data(pil_img, config=cfg, output_type=pytesseract.Output.DICT)
            conf = [int(c) for c in data['conf'] if c != '-1']
            avg_conf = float(np.mean(conf)/100.0) if conf else 0.0
            return txt, avg_conf
        except Exception:
            return "", 0.0

if uploaded:
    image = Image.open(uploaded).convert('RGB')
    orig_np = np.array(image)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Original")
        st.image(image, caption=f"Original ({orig_np.shape[1]}x{orig_np.shape[0]})", use_container_width=True)
    # Restoration
    if st.session_state.restoration_model is not None:
        restored_np = restore_image(orig_np)
    else:
        restored_np = orig_np.copy()
    with col_b:
        st.subheader("Restored")
        st.image(restored_np, caption=f"Restored ({restored_np.shape[1]}x{restored_np.shape[0]})", use_container_width=True)

    # Optional preprocessing preview
    processed_np = preprocessor.preprocess(restored_np)
    if show_preprocess:
        st.image(processed_np if processed_np.ndim==3 else processed_np, caption="Preprocessed", use_container_width=True)

    # OCR
    with st.spinner("Running Tesseract OCR..."):
        text, conf = ocr_multipass(processed_np, TES_LANG, multi=multi_pass)

    # Basic post-process
    def clean(txt: str) -> str:
        lines = [l.strip() for l in txt.split('\n') if l.strip()]
        return ' '.join(lines)
    cleaned = clean(text)
    devanagari_chars = sum(1 for c in cleaned if '\u0900' <= c <= '\u097F')

    st.markdown("### 📜 Extracted Sanskrit Text")
    if cleaned:
        st.text_area("Sanskrit (Devanagari)", cleaned, height=250)
    else:
        st.warning("No text extracted. Try different scan or enable multi-pass.")

    st.markdown("### 📈 OCR Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Confidence", f"{conf*100:.1f}%")
    with col2:
        st.metric("Chars", len(cleaned))
    with col3:
        st.metric("Devanagari Chars", devanagari_chars)

    # Download buttons
    st.markdown("### 💾 Downloads")
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button("⬇️ Text .txt", cleaned.encode('utf-8'), file_name="sanskrit_text.txt", mime="text/plain")
    with dl_col2:
        import json
        meta = {
            'confidence': conf,
            'language_used': TES_LANG,
            'chars_total': len(cleaned),
            'chars_devanagari': devanagari_chars,
        }
        st.download_button("⬇️ Metadata JSON", json.dumps(meta, ensure_ascii=False, indent=2).encode('utf-8'), file_name="ocr_meta.json", mime="application/json")
else:
    st.info("Upload an image to begin.")
    st.markdown("**Tip:** Use high-resolution scans for better restoration and OCR accuracy.")

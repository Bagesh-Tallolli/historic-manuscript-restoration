# ✅ Quick Start Checklist

Follow these steps to install and run the Sanskrit Manuscript Restoration Pipeline.

## Prerequisites
- [ ] Python 3.8 or higher installed
- [ ] Git installed
- [ ] 8GB+ disk space available
- [ ] (Optional) CUDA-capable GPU for faster training

---

## Installation Steps

### 1. Clone Repository
```bash
git clone https://github.com/Bagesh-Tallolli/historic-manuscript-restoration.git
cd historic-manuscript-restoration
```
- [ ] Repository cloned successfully

### 2. Create Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
- [ ] Virtual environment created
- [ ] Virtual environment activated

### 3. Upgrade pip
```bash
pip install --upgrade pip
```
- [ ] pip upgraded

### 4. Install Python Dependencies
**Option A - Manual:**
```bash
pip install -r requirements.txt
```

**Option B - Automated Script:**
```bash
./install.sh
```
- [ ] Dependencies installed without errors
- [ ] No dependency conflicts reported

### 5. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-san tesseract-ocr-hin
```

**macOS:**
```bash
brew install tesseract tesseract-lang
```

**Windows:**
- Download from: https://github.com/UB-Mannheim/tesseract/wiki
- Install and add to PATH

- [ ] Tesseract OCR installed
- [ ] Sanskrit language pack installed

### 6. Verify Installation
```bash
python verify_installation.py
```
- [ ] All checks pass (✓)
- [ ] No critical errors

---

## Quick Tests

### Test 1: Import Check
```python
python3 << EOF
from deep_translator import GoogleTranslator
from roboflow import Roboflow
print("✓ All imports successful!")
EOF
```
- [ ] Imports work without errors

### Test 2: Translation Module
```python
python3 << EOF
from nlp.translation import SanskritTranslator
translator = SanskritTranslator(method='google')
print("✓ Translation module ready!")
EOF
```
- [ ] Translation module loads correctly

### Test 3: Run Setup Test
```bash
python test_setup.py
```
- [ ] All module tests pass

---

## Next Steps

### Option 1: Process a Manuscript Image
```bash
python main.py --image_path path/to/manuscript.jpg
```
- [ ] Pipeline runs successfully

### Option 2: Train a Model
```bash
# Add training images to data/raw/
python train.py --epochs 50 --batch_size 8
```
- [ ] Training starts without errors

### Option 3: Start Web Interface
```bash
streamlit run app.py
```
- [ ] Web app launches successfully

### Option 4: Use Jupyter Notebook
```bash
jupyter notebook demo.ipynb
```
- [ ] Notebook opens and runs

---

## Troubleshooting

### Issue: Dependency conflicts
- [ ] Run: `pip cache purge`
- [ ] Reinstall: `pip install -r requirements.txt`
- [ ] See: INSTALLATION_GUIDE.md

### Issue: Tesseract not found
- [ ] Verify installation: `tesseract --version`
- [ ] Check PATH includes Tesseract
- [ ] See: INSTALLATION_GUIDE.md section "Tesseract OCR"

### Issue: Import errors
- [ ] Activate virtual environment
- [ ] Verify all packages: `python verify_installation.py`
- [ ] Check Python version: `python3 --version`

### Issue: Out of disk space
- [ ] Free up space
- [ ] Run: `pip cache purge`
- [ ] Install with: `pip install --no-cache-dir -r requirements.txt`

---

## Documentation Resources

- [ ] Read SOLUTION_REPORT.md for fix details
- [ ] Review INSTALLATION_GUIDE.md for detailed instructions
- [ ] Check FIX_SUMMARY.md for quick reference
- [ ] See README.md for project overview
- [ ] Explore GETTING_STARTED.md for usage examples

---

## Success Indicators

You know everything is working when:

✅ `pip install -r requirements.txt` completes without errors  
✅ `python verify_installation.py` shows all ✓  
✅ `python test_setup.py` passes all tests  
✅ No import errors when running Python scripts  
✅ `python main.py --image_path <file>` processes images successfully  

---

## Need Help?

1. **Check Documentation**: Start with INSTALLATION_GUIDE.md
2. **Run Verification**: `python verify_installation.py`
3. **Check Logs**: Look in logs/ directory for error details
4. **Test Setup**: Run `python test_setup.py`

---

**Once all checkboxes are marked, you're ready to use the pipeline! 🎉**

---

*For the complete solution to the dependency issue, see SOLUTION_REPORT.md*

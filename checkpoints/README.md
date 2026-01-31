# Model Checkpoints Directory

This directory contains trained model checkpoint files for the Historic Manuscript Restoration project.

## 📁 Directory Structure

```
checkpoints/
├── README.md                 ← You are here
├── latest_model.pth         ← Symlink to the latest/best model
└── kaggle/                  ← Models trained on Kaggle
    ├── desti.pth           ← Destination checkpoint
    ├── final.pth           ← Final trained model
    └── model_registry.json ← Model metadata (optional)
```

## 🎯 Quick Start

### 1. Place Your Trained Models Here

After training on Kaggle, download your `.pth` files and place them in `checkpoints/kaggle/`:

```bash
# Copy from Downloads
cp ~/Downloads/desti.pth checkpoints/kaggle/
cp ~/Downloads/final.pth checkpoints/kaggle/

# Or use the setup script
bash setup_kaggle_models.sh ~/Downloads
```

### 2. Test Your Models

```bash
python test_kaggle_models.py
```

### 3. Use for Inference

```bash
python inference.py \
    --input data/datasets/samples/ \
    --output output/restored \
    --checkpoint checkpoints/kaggle/final.pth \
    --model_size base
```

## 📋 Model File Information

### What's in a `.pth` file?

A PyTorch checkpoint file typically contains:
- `model_state_dict`: Model weights and parameters
- `optimizer_state_dict`: Optimizer state (for resuming training)
- `epoch`: Training epoch number
- `loss`: Final training loss
- `metrics`: Performance metrics (PSNR, SSIM, etc.)

### Expected File Sizes

| Model Size | Approximate Size |
|------------|------------------|
| Tiny       | ~50 MB           |
| Small      | ~100 MB          |
| Base       | ~200-300 MB      |
| Large      | ~500+ MB         |

## 🔧 Model Usage Examples

### Use in Python Script

```python
import torch
from models.vit_restorer import create_vit_restorer

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = create_vit_restorer('base', img_size=256)
checkpoint = torch.load('checkpoints/kaggle/final.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Use for inference
with torch.no_grad():
    output = model(input_tensor)
```

### Use with Streamlit App

The app will automatically look for models in this directory.

```bash
streamlit run app.py
```

### Use with API Server

```bash
python api_server.py --checkpoint checkpoints/kaggle/final.pth
```

## 📊 Model Registry (Optional)

You can track your models with a JSON registry:

```json
{
  "models": [
    {
      "name": "final.pth",
      "training_date": "2025-11-24",
      "epochs": 100,
      "dataset": "roboflow_sanskrit",
      "architecture": "ViT-Base",
      "img_size": 256,
      "batch_size": 16,
      "learning_rate": 0.0001,
      "metrics": {
        "psnr": 28.5,
        "ssim": 0.85,
        "val_loss": 0.025
      },
      "notes": "Best performing model on validation set"
    }
  ]
}
```

## 🔍 Troubleshooting

### Model won't load?

1. Check file exists: `ls -lh checkpoints/kaggle/*.pth`
2. Verify file size (should be > 50 MB)
3. Test with: `python test_kaggle_models.py`
4. Check model size matches: use `--model_size base` if trained with base

### Out of memory?

- Use CPU: `--device cpu`
- Reduce batch size
- Process images one at a time

### Wrong architecture?

Make sure you're using the same model size you trained with:
- `--model_size tiny` for tiny models
- `--model_size small` for small models  
- `--model_size base` for base models (default)
- `--model_size large` for large models

## 🚀 Next Steps

1. ✅ Download models from Kaggle
2. ✅ Place in `checkpoints/kaggle/`
3. ✅ Run `python test_kaggle_models.py`
4. ✅ Test inference on sample images
5. ✅ Deploy to production!

## 📚 See Also

- `KAGGLE_MODEL_INTEGRATION.md` - Detailed integration guide
- `inference.py` - Inference script
- `app.py` - Streamlit application
- `api_server.py` - REST API server


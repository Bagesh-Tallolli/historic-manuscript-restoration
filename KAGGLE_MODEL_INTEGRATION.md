# Kaggle Model Integration Guide

## 📁 Where to Place Your Trained Model Files

After training your model on Kaggle, you'll have checkpoint files (`.pth` files). Here's where to place them:

### Recommended Directory Structure:

```
EL-project/
├── checkpoints/
│   └── kaggle/                    ← Place your Kaggle-trained models here
│       ├── desti.pth             ← Your destination/final checkpoint
│       ├── final.pth             ← Your final model checkpoint
│       ├── best_model.pth        ← Best performing checkpoint (optional)
│       └── epoch_XX.pth          ← Any intermediate checkpoints (optional)
│
├── models/
│   └── trained_models/            ← Alternative location for production models
│       ├── desti.pth
│       └── final.pth
```

## 📥 Step-by-Step: Download from Kaggle

### Option 1: Download via Kaggle Notebook (Recommended)

Add this cell at the end of your Kaggle training notebook:

```python
# After training completes, download your models
from IPython.display import FileLink

# Create download links
print("Download your trained models:")
display(FileLink('desti.pth'))
display(FileLink('final.pth'))
```

### Option 2: Use Kaggle API

```bash
# Install Kaggle CLI
pip install kaggle

# Download your notebook's output files
kaggle kernels output <your-username>/<notebook-name> -p ./checkpoints/kaggle/
```

### Option 3: Manual Download

1. In your Kaggle notebook, go to the output section on the right panel
2. Click on the file (desti.pth or final.pth)
3. Download to your local machine
4. Move to the appropriate directory

## 🔧 How to Use Your Trained Models

### 1. For Inference (Restoration)

Place your model in `checkpoints/kaggle/` and run:

```bash
# Using desti.pth
python inference.py \
    --input data/raw/test/your_image.jpg \
    --output output/restored \
    --checkpoint checkpoints/kaggle/desti.pth \
    --model_size base \
    --img_size 256

# Using final.pth
python inference.py \
    --input data/raw/test/ \
    --output output/restored \
    --checkpoint checkpoints/kaggle/final.pth \
    --model_size base \
    --img_size 256
```

### 2. For the Streamlit App

Update `app.py` to use your Kaggle model:

```python
# In app.py, modify the checkpoint path
checkpoint_path = "checkpoints/kaggle/final.pth"  # or "checkpoints/kaggle/desti.pth"
```

Then run:
```bash
bash run_app.sh
# or
streamlit run app.py
```

### 3. For API Server

```bash
# Start the API server with your model
python api_server.py --checkpoint checkpoints/kaggle/final.pth
```

## 📋 File Commands Quick Reference

### Copy files to the correct location:

```bash
# If you downloaded to Downloads folder
cp ~/Downloads/desti.pth /home/bagesh/EL-project/checkpoints/kaggle/
cp ~/Downloads/final.pth /home/bagesh/EL-project/checkpoints/kaggle/

# Or use mv to move instead of copy
mv ~/Downloads/desti.pth /home/bagesh/EL-project/checkpoints/kaggle/
mv ~/Downloads/final.pth /home/bagesh/EL-project/checkpoints/kaggle/
```

### Verify files are in place:

```bash
ls -lh /home/bagesh/EL-project/checkpoints/kaggle/
```

## 🔍 Understanding Your Model Files

### desti.pth
- **Purpose**: Destination/final checkpoint from training
- **Contains**: Model weights, optimizer state, epoch info
- **Use for**: Production inference, deployment

### final.pth
- **Purpose**: Final model checkpoint after training completion
- **Contains**: Complete training state
- **Use for**: Best results, continued training

## 🧪 Test Your Model

After placing the files, test them:

```bash
# Quick test with a sample image
python inference.py \
    --input data/datasets/samples/ \
    --output output/test_kaggle_model \
    --checkpoint checkpoints/kaggle/final.pth \
    --model_size base
```

## 🔄 Model File Format

Your `.pth` files should contain:

```python
{
    'epoch': <epoch_number>,
    'model_state_dict': <model_weights>,
    'optimizer_state_dict': <optimizer_state>,
    'loss': <final_loss>,
    'metrics': <performance_metrics>
}
```

To check what's in your model file:

```python
import torch

checkpoint = torch.load('checkpoints/kaggle/final.pth', map_location='cpu')
print("Keys in checkpoint:", checkpoint.keys())
print("Epoch:", checkpoint.get('epoch', 'N/A'))
print("Loss:", checkpoint.get('loss', 'N/A'))
```

## 🚨 Troubleshooting

### Error: "File not found"
- **Solution**: Verify the file path with `ls checkpoints/kaggle/`
- Make sure you're in the project root directory

### Error: "Model architecture mismatch"
- **Solution**: Ensure you're using the same `--model_size` that you trained with
- Check if the model was trained with `img_size=256` (default)

### Error: "CUDA out of memory"
- **Solution**: Use `--device cpu` for inference on CPU
- Or process images one at a time

### Large file size?
- **Normal**: Model files can be 50-500 MB depending on architecture
- **Compress**: Use `gzip` for storage: `gzip checkpoints/kaggle/final.pth`
- **Decompress**: `gunzip checkpoints/kaggle/final.pth.gz` before use

## 📊 Model Performance Tracking

Create a model registry:

```bash
# Create a JSON file to track your models
cat > checkpoints/kaggle/model_registry.json << 'EOF'
{
  "models": [
    {
      "name": "final.pth",
      "training_date": "2025-11-24",
      "epochs": 100,
      "dataset": "roboflow_sanskrit",
      "psnr": 28.5,
      "ssim": 0.85,
      "notes": "Best overall performance"
    },
    {
      "name": "desti.pth",
      "training_date": "2025-11-24",
      "epochs": 100,
      "dataset": "roboflow_sanskrit",
      "notes": "Destination checkpoint"
    }
  ]
}
EOF
```

## 🎯 Next Steps

1. ✅ Download `desti.pth` and `final.pth` from Kaggle
2. ✅ Place them in `checkpoints/kaggle/` directory
3. ✅ Test with `inference.py`
4. ✅ Update `app.py` to use your model
5. ✅ Run the Streamlit app to see results
6. ✅ Share your trained model or deploy to production!

---

**Need Help?** Check the model loading in `inference.py` or `app.py` for examples.


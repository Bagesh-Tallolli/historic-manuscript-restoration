# KAGGLE DATASET DOWNLOAD - COMPLETE EXAMPLES

This guide shows you exactly how to configure each dataset download method in your Kaggle notebook.

---

## 🎯 Quick Summary

The training script supports **6 different ways** to get your dataset:
1. **Roboflow** - Best for ML datasets with annotations
2. **Kaggle Dataset** - Public datasets from Kaggle
3. **URL Download** - Direct link to zip file
4. **Google Drive** - Your personal datasets
5. **Kaggle Input** - Manually uploaded via UI
6. **Sample Dataset** - Auto-generated test data

---

## METHOD 1: Roboflow (RECOMMENDED) 🏆

### Why Use This?
- Professional manuscript datasets
- Pre-organized train/val/test splits
- Easy version control
- Great for Sanskrit/historical documents

### Step-by-Step Setup:

#### 1. Get Your Roboflow API Key
```
1. Go to https://app.roboflow.com
2. Sign up (free account works!)
3. Click your profile picture (top-right)
4. Select "Roboflow API"
5. Copy your private API key
```

#### 2. Find Your Dataset Details
```
In Roboflow, your dataset URL looks like:
https://app.roboflow.com/WORKSPACE/PROJECT/VERSION

Example:
https://app.roboflow.com/my-manuscripts/sanskrit-docs/1

Extract:
- WORKSPACE: my-manuscripts
- PROJECT: sanskrit-docs
- VERSION: 1
```

#### 3. Configure in Kaggle Notebook
```python
# In main() function, set these:
USE_ROBOFLOW = True
ROBOFLOW_CONFIG = {
    'api_key': 'YOUR_ACTUAL_API_KEY_HERE',
    'workspace': 'my-manuscripts',
    'project': 'sanskrit-docs',
    'version': 1
}

# Set all others to False
USE_KAGGLE_DATASET = False
USE_URL_DOWNLOAD = False
USE_GOOGLE_DRIVE = False
USE_KAGGLE_INPUT = False
USE_SAMPLE_DATASET = False
```

#### 4. Run!
The dataset will automatically download to `/kaggle/working/dataset/`

---

## METHOD 2: Kaggle Dataset

### Why Use This?
- Access thousands of public datasets
- No API key needed
- Fast downloads
- Community datasets

### Step-by-Step Setup:

#### 1. Find a Dataset
```
1. Go to https://www.kaggle.com/datasets
2. Search for "manuscripts" or "documents"
3. Click on a dataset
4. Copy the dataset identifier from URL

Example URL:
https://www.kaggle.com/datasets/johnsmith/historical-manuscripts

Dataset name: johnsmith/historical-manuscripts
```

#### 2. Setup Kaggle API (ONE-TIME)
```python
# In a Kaggle notebook cell, run:
import os
os.makedirs('/root/.kaggle', exist_ok=True)

# Option A: If you have kaggle.json
# Upload kaggle.json to notebook, then:
!mv kaggle.json /root/.kaggle/
!chmod 600 /root/.kaggle/kaggle.json

# Option B: Manual setup
with open('/root/.kaggle/kaggle.json', 'w') as f:
    f.write('{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}')
!chmod 600 /root/.kaggle/kaggle.json
```

To get your API key:
```
1. Go to kaggle.com/USERNAME/account
2. Scroll to "API" section
3. Click "Create New API Token"
4. Download kaggle.json
```

#### 3. Configure in Notebook
```python
USE_KAGGLE_DATASET = True
KAGGLE_DATASET_NAME = 'johnsmith/historical-manuscripts'  # ← Your dataset

# Set all others to False
USE_ROBOFLOW = False
USE_URL_DOWNLOAD = False
USE_GOOGLE_DRIVE = False
USE_KAGGLE_INPUT = False
USE_SAMPLE_DATASET = False
```

---

## METHOD 3: Direct URL Download

### Why Use This?
- Download from any public URL
- Works with university archives
- Direct control over source
- No API needed

### Step-by-Step Setup:

#### 1. Get Your Dataset URL
Your dataset must be:
- A direct link to a ZIP file
- Publicly accessible (no login required)
- Contains train/ folder with images

Example URLs:
```
https://example.com/datasets/manuscripts.zip
https://storage.university.edu/historical-data.zip
https://archive.org/download/ancient-texts/dataset.zip
```

#### 2. Configure in Notebook
```python
USE_URL_DOWNLOAD = True
DATASET_URL = 'https://example.com/datasets/manuscripts.zip'  # ← Your URL

# Set all others to False
USE_ROBOFLOW = False
USE_KAGGLE_DATASET = False
USE_GOOGLE_DRIVE = False
USE_KAGGLE_INPUT = False
USE_SAMPLE_DATASET = False
```

#### 3. Run!
The script will:
- Download the ZIP file
- Extract it
- Auto-detect train/val folders

---

## METHOD 4: Google Drive

### Why Use This?
- Host your own datasets
- Large file support (up to 15GB free)
- Private datasets
- Easy sharing

### Step-by-Step Setup:

#### 1. Upload Dataset to Google Drive
```
1. Create a ZIP file with your dataset:
   dataset.zip
   └── train/
       ├── img1.jpg
       ├── img2.jpg
       └── ...

2. Upload to Google Drive
3. Right-click the file → Share → "Anyone with the link can view"
4. Copy the share link
```

#### 2. Extract File ID
```
Your share link looks like:
https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0j/view?usp=sharing

File ID is the middle part:
1a2b3c4d5e6f7g8h9i0j
```

#### 3. Configure in Notebook
```python
USE_GOOGLE_DRIVE = True
GOOGLE_DRIVE_FILE_ID = '1a2b3c4d5e6f7g8h9i0j'  # ← Your file ID

# Set all others to False
USE_ROBOFLOW = False
USE_KAGGLE_DATASET = False
USE_URL_DOWNLOAD = False
USE_KAGGLE_INPUT = False
USE_SAMPLE_DATASET = False
```

#### 4. Run!
Downloads via `gdown` package (installed automatically)

---

## METHOD 5: Kaggle Input (Manual Upload)

### Why Use This?
- Traditional Kaggle method
- Visual interface
- Pre-verified datasets
- No code needed for upload

### Step-by-Step Setup:

#### 1. Create Kaggle Dataset
```
1. Go to https://www.kaggle.com/datasets
2. Click "New Dataset"
3. Upload your images folder
4. Name it (e.g., "my-manuscript-dataset")
5. Click "Create"
```

#### 2. Add to Notebook
```
1. In your Kaggle notebook, click "Add Data" (right sidebar)
2. Search for your dataset name
3. Click "Add"
```

#### 3. Find the Path
```
After adding, check the path:
/kaggle/input/YOUR-DATASET-NAME/

List contents:
!ls /kaggle/input/YOUR-DATASET-NAME/
```

#### 4. Configure in Notebook
```python
USE_KAGGLE_INPUT = True
TRAIN_DIR = '/kaggle/input/my-manuscript-dataset/train'  # ← Your path
VAL_DIR = '/kaggle/input/my-manuscript-dataset/val'      # Optional

# Set all others to False
USE_ROBOFLOW = False
USE_KAGGLE_DATASET = False
USE_URL_DOWNLOAD = False
USE_GOOGLE_DRIVE = False
USE_SAMPLE_DATASET = False
```

---

## METHOD 6: Sample Dataset (Testing)

### Why Use This?
- No dataset needed!
- Quick testing
- Verify training works
- Learn the system

### Step-by-Step Setup:

#### 1. Just Enable It!
```python
USE_SAMPLE_DATASET = True

# Set all others to False
USE_ROBOFLOW = False
USE_KAGGLE_DATASET = False
USE_URL_DOWNLOAD = False
USE_GOOGLE_DRIVE = False
USE_KAGGLE_INPUT = False
```

#### 2. Run!
Creates 50 synthetic "manuscript-like" images automatically.

**Note:** These are random images, not real manuscripts. Use only for testing!

---

## 🔍 Complete Example Configurations

### Example 1: Roboflow Dataset
```python
def main():
    # ========== DATASET CONFIGURATION ==========
    USE_ROBOFLOW = True
    ROBOFLOW_CONFIG = {
        'api_key': 'abc123xyz789',
        'workspace': 'university-archive',
        'project': 'sanskrit-manuscripts',
        'version': 2
    }
    
    USE_KAGGLE_DATASET = False
    USE_URL_DOWNLOAD = False
    USE_GOOGLE_DRIVE = False
    USE_KAGGLE_INPUT = False
    USE_SAMPLE_DATASET = False
    
    # ... rest of main() ...
```

### Example 2: Google Drive + Custom Settings
```python
def main():
    # ========== DATASET CONFIGURATION ==========
    USE_GOOGLE_DRIVE = True
    GOOGLE_DRIVE_FILE_ID = '1a2b3c4d5e6f7g8h9i0j'
    
    USE_ROBOFLOW = False
    USE_KAGGLE_DATASET = False
    USE_URL_DOWNLOAD = False
    USE_KAGGLE_INPUT = False
    USE_SAMPLE_DATASET = False
    
    # ========== TRAINING CONFIGURATION ==========
    IMG_SIZE = 256
    BATCH_SIZE = 8  # Reduced for memory
    NUM_EPOCHS = 50
    MODEL_SIZE = 'small'  # Smaller model
    
    # ... rest of main() ...
```

### Example 3: Quick Test with Sample Data
```python
def main():
    # ========== DATASET CONFIGURATION ==========
    USE_SAMPLE_DATASET = True
    
    USE_ROBOFLOW = False
    USE_KAGGLE_DATASET = False
    USE_URL_DOWNLOAD = False
    USE_GOOGLE_DRIVE = False
    USE_KAGGLE_INPUT = False
    
    # ========== TRAINING CONFIGURATION ==========
    NUM_EPOCHS = 10  # Just test for 10 epochs
    MODEL_SIZE = 'tiny'  # Fastest model
    
    # ... rest of main() ...
```

---

## 🛠️ Troubleshooting

### Roboflow Issues

**Error: "Invalid API key"**
- Check you copied the entire key
- Make sure it's your private key, not public
- Try regenerating the key

**Error: "Project not found"**
- Verify workspace/project names exactly match
- Check capitalization
- Make sure version exists

### Kaggle Dataset Issues

**Error: "Dataset not found"**
- Make sure format is: `username/dataset-name`
- Check dataset is public or you have access
- Verify kaggle.json is in /root/.kaggle/

**Error: "Unauthorized"**
- Re-download kaggle.json
- Check file permissions: `chmod 600 /root/.kaggle/kaggle.json`

### Google Drive Issues

**Error: "Cannot download file"**
- Make sure sharing is "Anyone with the link"
- Check file ID is correct (just the ID, not full URL)
- Try re-sharing the file

**Error: "File too large"**
- gdown has limits on very large files
- Consider splitting into smaller zips
- Or use Kaggle Dataset method instead

### URL Download Issues

**Error: "Connection failed"**
- Check URL is accessible (try in browser)
- Make sure it's a direct link to .zip
- Some sites block automated downloads

---

## 💡 Best Practices

1. **For Production**: Use Roboflow or Kaggle Dataset
2. **For Personal Data**: Use Google Drive
3. **For Testing**: Use Sample Dataset
4. **For Public Archives**: Use URL Download
5. **For Maximum Control**: Use Kaggle Input

---

## 📊 Comparison Table

| Method | Setup Time | Best For | Requires API | Size Limit |
|--------|-----------|----------|--------------|------------|
| Roboflow | 5 min | ML datasets | Yes | 10GB free |
| Kaggle Dataset | 2 min | Public data | Yes | No limit |
| URL Download | 1 min | Archives | No | Depends |
| Google Drive | 3 min | Personal data | No | 15GB free |
| Kaggle Input | 10 min | Manual control | No | 20GB |
| Sample | 0 min | Testing | No | N/A |

---

## ✅ Quick Checklist

Before running training:
- [ ] Chose ONE dataset method
- [ ] Set corresponding USE_* flag to True
- [ ] Set all other USE_* flags to False
- [ ] Filled in required credentials/paths
- [ ] Enabled GPU in Kaggle settings
- [ ] Enabled Internet in Kaggle settings
- [ ] Installed all dependencies

---

## 🚀 Ready to Go!

Pick your method, configure it, and run! The script handles everything else automatically.

**Need help?** Check the error message and refer to the troubleshooting section above.


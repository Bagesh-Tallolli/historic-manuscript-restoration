#!/usr/bin/env python
# coding: utf-8

# # 🏛️ Historic Manuscript Restoration with Vision Transformer\n\n## 📋 Overview\nThis notebook trains a Vision Transformer (ViT) model for restoring degraded historic manuscripts.\n\n## ⚙️ Setup Instructions\n1. **Enable GPU**: Settings → Accelerator → GPU T4 x2\n2. **Run all cells**: Dataset downloads automatically from Roboflow\n3. **Wait for training**: ~5 hours on GPU T4\n4. **Download model**: Best model saved to `/kaggle/working/checkpoints/best_psnr.pth`\n\n## 🎯 Features\n- ✅ Automatic dataset download from Roboflow\n- ✅ Synthetic degradation pipeline\n- ✅ Vision Transformer architecture\n- ✅ Multi-metric evaluation (PSNR, SSIM, LPIPS)\n- ✅ Automatic checkpointing\n\n---

# ## 1️⃣ Install Dependencies

# In[ ]:


get_ipython().run_cell_magic('capture\\n!pip', 'install einops lpips roboflow opencv-python-headless scikit-image -q', '')


# ## 2️⃣ Import Libraries

# In[ ]:


import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nimport torch.optim as optim\nfrom torch.utils.data import Dataset, DataLoader\nfrom pathlib import Path\nimport cv2\nimport numpy as np\nfrom PIL import Image\nimport random\nfrom tqdm import tqdm\nimport json\nimport time\nimport os\nfrom einops import rearrange\nfrom einops.layers.torch import Rearrange\nimport lpips\nfrom skimage.metrics import peak_signal_noise_ratio, structural_similarity\n\nprint(f\"✓ PyTorch version: {torch.__version__}\")\nprint(f\"✓ CUDA available: {torch.cuda.is_available()}\")\nif torch.cuda.is_available():\n    print(f\"✓ GPU: {torch.cuda.get_device_name(0)}\")


# ## 3️⃣ Download Dataset from Roboflow\n\n**Dataset**: Sanskrit Manuscript Dataset  \n**Source**: Roboflow (preconfigured)\n\nThe dataset will be automatically downloaded and prepared for training.

# In[ ]:


from roboflow import Roboflow# ========== ROBOFLOW CONFIGURATION (HARDCODED) ==========ROBOFLOW_API_KEY = "EBJvHlgSWLyW1Ir6ctkH"ROBOFLOW_WORKSPACE = "neeew"ROBOFLOW_PROJECT = "yoyoyo-mptyx-ijqfp"ROBOFLOW_VERSION = 1DATASET_LOCATION = "/kaggle/working/dataset"print("=" * 70)print("📥 DOWNLOADING DATASET FROM ROBOFLOW")print("=" * 70)print(f"Workspace: {ROBOFLOW_WORKSPACE}")print(f"Project: {ROBOFLOW_PROJECT}")print(f"Version: {ROBOFLOW_VERSION}")print(f"Download location: {DATASET_LOCATION}\n")try:    # Initialize Roboflow    rf = Roboflow(api_key=ROBOFLOW_API_KEY)        # Get project and download    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)    dataset = project.version(ROBOFLOW_VERSION).download("folder", location=DATASET_LOCATION)        print("\n✅ Dataset downloaded successfully!")    print(f"📁 Location: {DATASET_LOCATION}")        # Check directory structure    for subdir in ['train', 'valid', 'test']:        path = Path(DATASET_LOCATION) / subdir        if path.exists():            num_files = len(list(path.glob('*.*')))            print(f"  ✓ {subdir}: {num_files} files")        # Set train and validation directories    TRAIN_DIR = f"{DATASET_LOCATION}/train"    VAL_DIR = f"{DATASET_LOCATION}/valid" if os.path.exists(f"{DATASET_LOCATION}/valid") else None    except Exception as e:    print(f"\n❌ Error downloading dataset: {e}")    print("\nPlease check:")    print("1. API key is correct")    print("2. Internet connection is working")    print("3. Roboflow project exists and is accessible")    raise


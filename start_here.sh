#!/bin/bash
# One-command starter for Roboflow dataset

cat << "EOF"

╔═══════════════════════════════════════════════════════════════════╗
║                  🚀 ROBOFLOW DATASET - QUICK START                ║
╔═══════════════════════════════════════════════════════════════════╗

Dataset: https://universe.roboflow.com/sanskritocr/yoyoyo-mptyx

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 WHAT YOU NEED:

1. Roboflow account (free): https://app.roboflow.com/
2. API key from: https://app.roboflow.com/settings/api
3. ~5-10 minutes

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚡ QUICK START:

   bash setup_roboflow.sh

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 DOCUMENTATION:

   • Complete guide:    cat ROBOFLOW_SETUP.md
   • Quick reference:   cat QUICK_REFERENCE.txt
   • Workflow diagram:  cat WORKFLOW_DIAGRAM.txt

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ VERIFY STATUS:

   bash check_dataset.sh

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎓 START TRAINING:

   python3 train.py --train_dir data/raw/train --val_dir data/raw/val

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Ready to download the dataset? Run:  bash setup_roboflow.sh

╚═══════════════════════════════════════════════════════════════════╝

EOF

echo ""
read -p "Download dataset now? (y/n): " choice

if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
    bash setup_roboflow.sh
else
    echo ""
    echo "No problem! When ready, run:"
    echo "  bash setup_roboflow.sh"
    echo ""
fi


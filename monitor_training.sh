#!/bin/bash
# Monitor training progress

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                    TRAINING PROGRESS MONITOR                                 ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if training is running
if pgrep -f "train.py" > /dev/null; then
    echo "✅ Training process is RUNNING"
    echo ""
else
    echo "⚠️  Training process is NOT running"
    echo ""
fi

# Show dataset info
echo "📊 Dataset:"
echo "   Training:   $(ls data/raw/train/*.jpg 2>/dev/null | wc -l) images"
echo "   Validation: $(ls data/raw/val/*.jpg 2>/dev/null | wc -l) images"
echo ""

# Show latest log output
echo "📝 Latest Training Output:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f "training_output.log" ]; then
    tail -30 training_output.log
else
    echo "No log file found yet"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check for checkpoints
echo "💾 Checkpoints:"
if [ -d "models/checkpoints" ]; then
    ls -lth models/checkpoints/ | head -10
else
    echo "   No checkpoints yet"
fi
echo ""

echo "🔄 To monitor in real-time: tail -f training_output.log"
echo "🛑 To stop training: pkill -f train.py"


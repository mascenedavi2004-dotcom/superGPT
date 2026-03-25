#!/bin/bash
# =============================================================================
#  superGPT — One-Shot RunPod Training Script
#  
#  Downloads FineWeb-Edu, prepares 100M tokens with quality filtering,
#  and trains the model. Run this on a RunPod GPU pod.
#
#  Budget: ~$2-3 total (2-3 hours on RTX A4500 @ $0.26/hr)
#  Expected: val loss ~2.5-3.0, coherent English text generation
#
#  Usage:
#    chmod +x scripts/runpod_train.sh
#    ./scripts/runpod_train.sh
# =============================================================================

set -e

echo "============================================================"
echo "  superGPT — Production Training Pipeline"
echo "  Dataset: FineWeb-Edu (high-quality educational text)"
echo "============================================================"

# ── Setup ────────────────────────────────────────────────────────────────────
cd /workspace

# Clone if not already present
if [ ! -d "superGPT" ]; then
    echo "Cloning superGPT..."
    git clone https://github.com/viralcode/superGPT.git
    cd superGPT
else
    cd superGPT
    echo "Pulling latest..."
    git pull origin main
fi

# Install dependencies
echo "Installing dependencies..."
pip install -q torch numpy transformers datasets tokenizers

# Set CUDA memory config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Step 1: Prepare Data ─────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Step 1: Downloading & Preparing FineWeb-Edu"
echo "  Target: 100M tokens with quality filtering + dedup"
echo "============================================================"

python -u -m supergpt.training.data_pipeline \
    --dataset HuggingFaceFW/fineweb-edu \
    --subset sample-10BT \
    --tokenizer Qwen/Qwen2.5-0.5B \
    --output-dir data \
    --max-tokens 100000000 \
    --shard-size 10000000

echo ""
echo "Data preparation complete!"
ls -lh data/

# ── Step 2: Train ────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Step 2: Training (small preset, ~69M params)"
echo "  Expected time: ~2 hours on RTX A4500"
echo "============================================================"

python -m supergpt.training.train \
    --preset small \
    --data-dir data \
    --max-iters 15000 \
    --batch-size 16 \
    --lr 3e-4 \
    --device cuda \
    --lr-schedule cosine

# ── Step 3: Generate Text ────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Step 3: Text Generation Test"
echo "============================================================"

python scripts/generate.py \
    --checkpoint checkpoints/best.pt \
    --prompt "Artificial intelligence is" \
    --max-tokens 200 \
    --temperature 0.7

echo ""
python scripts/generate.py \
    --checkpoint checkpoints/best.pt \
    --prompt "The history of science began when" \
    --max-tokens 200 \
    --temperature 0.7

echo ""
python scripts/generate.py \
    --checkpoint checkpoints/best.pt \
    --prompt "In mathematics, the concept of" \
    --max-tokens 200 \
    --temperature 0.7

echo ""
echo "============================================================"
echo "  Training complete! Check outputs above."
echo "============================================================"

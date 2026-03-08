#!/bin/bash
set -e
export OMP_NUM_THREADS=1
cd ~/twm_diffusion_package

COMMON="--data-dir data/atomic_10000 \
    --config base \
    --denoiser-layers 1 --denoiser-dim 128 --denoiser-heads 4 \
    --n-proj-tokens 4 \
    --pretrained-dynamics pretrained/model_best.pt \
    --freeze-dynamics --freeze-encoder \
    --epochs 600 --batch-size 32 --lr 3e-4 \
    --patience 100 --log-every 10 --device cuda:1"

# Run 1: adaLN-Zero + cross-attention (4 memory tokens) — primary experiment
echo ">>> Run 1: v7_adaln (adaLN-Zero + xattn 4 tokens)"
uv run python scripts/train_diffusion_v7.py \
    --out-dir results/v7_adaln \
    --use-adaln \
    $COMMON

# Run 2: adaLN-Zero only, no cross-attention
echo ">>> Run 2: v7_adaln_only (adaLN-Zero, no xattn)"
uv run python scripts/train_diffusion_v7.py \
    --out-dir results/v7_adaln_only \
    --use-adaln --no-cross-attention \
    $COMMON

# Run 3: adaLN-Zero + cross-attention with 16 memory tokens
echo ">>> Run 3: v7_adaln_16mem (adaLN-Zero + xattn 16 tokens)"
uv run python scripts/train_diffusion_v7.py \
    --out-dir results/v7_adaln_16mem \
    --use-adaln --n-proj-tokens 16 \
    $COMMON

# Inference on best checkpoints
echo ">>> Inference sweeps"
for dir in results/v7_adaln results/v7_adaln_only results/v7_adaln_16mem; do
    if [ -f "$dir/model_best.pt" ]; then
        echo "  $dir: steps=10, temp=0.0"
        uv run python scripts/infer_diffusion_model.py \
            --model-dir "$dir" \
            --data-dir data/atomic_10000 \
            --denoise-steps 10 \
            --temperature 0.0 \
            --split test --n-examples 30 \
            --device cuda:1 2>&1 | tee "${dir}/infer_steps10_temp0.0.txt"
    fi
done

echo "=== All done ==="

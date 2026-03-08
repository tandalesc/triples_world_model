#!/bin/bash
set -e
export OMP_NUM_THREADS=1
cd ~/twm_diffusion_package

COMMON="--data-dir data/atomic_10000 \
    --config base \
    --denoiser-layers 1 --denoiser-dim 128 --denoiser-heads 4 \
    --pretrained-dynamics pretrained/model_best.pt \
    --freeze-dynamics --freeze-encoder \
    --mask-beta-a 1.0 --mask-beta-b 1.0 \
    --epochs 300 --batch-size 32 --lr 3e-4 \
    --patience 50 --log-every 10 --device cuda:1"

# Run 1: FiLM only (no cross-attention)
echo ">>> Run 1: FiLM only (no cross-attention)"
uv run python scripts/train_diffusion_v5.py \
    --out-dir results/v5_film_only \
    --use-film --no-cross-attention \
    --n-proj-tokens 4 \
    $COMMON

# Run 2: FiLM + cross-attention (4 memory tokens)
echo ">>> Run 2: FiLM + cross-attention (4 tokens)"
uv run python scripts/train_diffusion_v5.py \
    --out-dir results/v5_film_xattn4 \
    --use-film \
    --n-proj-tokens 4 \
    $COMMON

# Run 3: FiLM + cross-attention (16 memory tokens)
echo ">>> Run 3: FiLM + cross-attention (16 tokens)"
uv run python scripts/train_diffusion_v5.py \
    --out-dir results/v5_film_xattn16 \
    --use-film \
    --n-proj-tokens 16 \
    $COMMON

# Inference on best checkpoints
echo ">>> Inference sweeps"
for dir in results/v5_film_only results/v5_film_xattn4 results/v5_film_xattn16; do
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

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
    --patience 100 --log-every 10 --diagnostic-every 50 \
    --device cuda:1"

# Run 1: Cosine schedule + alpha_min + importance sampling (primary)
echo ">>> Run 1: v9_alphamin (cosine + alpha_min=0.01, bias_power=2)"
uv run python scripts/train_diffusion_v9.py \
    --out-dir results/v9_alphamin \
    --use-adaln --use-continuous-noise \
    --alpha-min 0.01 --timestep-bias-power 2.0 \
    $COMMON

# Run 2: Cosine schedule + no alpha_min (baseline, expect cliff at t=1.0)
echo ">>> Run 2: v9_baseline (cosine, no alpha_min, uniform sampling)"
uv run python scripts/train_diffusion_v9.py \
    --out-dir results/v9_baseline \
    --use-adaln --use-continuous-noise \
    $COMMON

# Inference on best checkpoints
echo ">>> Inference sweeps"
for dir in results/v9_alphamin results/v9_baseline; do
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

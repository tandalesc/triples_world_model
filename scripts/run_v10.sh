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
    --use-adaln --use-continuous-noise --unified-decoder \
    --alpha-min 0.01 --timestep-bias-power 2.0 \
    --device cuda:1"

# Run 1: Unified decoder initialized from v9 value decoder
echo ">>> Run 1: v10_unified (init from v9 value decoder)"
uv run python scripts/train_diffusion_v10.py \
    --out-dir results/v10_unified \
    --v9-checkpoint results/v9_alphamin/model_best.pt \
    $COMMON

# Run 2: Unified decoder from scratch
echo ">>> Run 2: v10_unified_scratch (random init)"
uv run python scripts/train_diffusion_v10.py \
    --out-dir results/v10_unified_scratch \
    $COMMON

# Inference on best checkpoints
echo ">>> Inference sweeps"
for dir in results/v10_unified results/v10_unified_scratch; do
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

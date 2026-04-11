#!/bin/bash
set -e
export OMP_NUM_THREADS=1
cd ~/twm_diffusion_package

V7_DIR="results/v7_adaln"
V7_CKPT="${V7_DIR}/model_best.pt"

COMMON="--data-dir data/atomic_10000 \
    --config base \
    --v7-model-dir ${V7_DIR} \
    --v7-checkpoint ${V7_CKPT} \
    --epochs 200 --batch-size 32 \
    --patience 50 --log-every 5 --diagnostic-every 50 \
    --device cuda:1"

# Run 1: Conditioning-only, 0.1x LR (3e-5)
echo ">>> Run 1: v8_pure1.0_condonly_lr0.1x"
uv run python scripts/train_diffusion_v8.py \
    --out-dir results/v8_pure1.0_condonly_lr0.1x \
    --conditioning-only --lr 3e-5 \
    $COMMON

# Run 2: Conditioning-only, 0.3x LR (9e-5)
echo ">>> Run 2: v8_pure1.0_condonly_lr0.3x"
uv run python scripts/train_diffusion_v8.py \
    --out-dir results/v8_pure1.0_condonly_lr0.3x \
    --conditioning-only --lr 9e-5 \
    $COMMON

# Run 3: Full fine-tune, 0.1x LR (3e-5)
echo ">>> Run 3: v8_pure1.0_full_lr0.1x"
uv run python scripts/train_diffusion_v8.py \
    --out-dir results/v8_pure1.0_full_lr0.1x \
    --lr 3e-5 \
    $COMMON

# Run 4: Full fine-tune, 0.3x LR (9e-5)
echo ">>> Run 4: v8_pure1.0_full_lr0.3x"
uv run python scripts/train_diffusion_v8.py \
    --out-dir results/v8_pure1.0_full_lr0.3x \
    --lr 9e-5 \
    $COMMON

# Inference on best checkpoints
echo ">>> Inference sweeps"
for dir in results/v8_pure1.0_condonly_lr0.1x results/v8_pure1.0_condonly_lr0.3x \
           results/v8_pure1.0_full_lr0.1x results/v8_pure1.0_full_lr0.3x; do
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

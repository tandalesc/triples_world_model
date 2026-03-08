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
    --epochs 300 --batch-size 32 --lr 3e-4 \
    --patience 50 --log-every 10 --device cuda:1"

# ===== Phase 1: Masking schedule ablation =====

echo ">>> Phase 1, Run 1: Beta(2,1)"
uv run python scripts/train_diffusion_model.py \
    --out-dir results/v3_beta2_1 \
    --mask-beta-a 2.0 --mask-beta-b 1.0 \
    $COMMON

echo ">>> Phase 1, Run 2: Beta(3,1)"
uv run python scripts/train_diffusion_model.py \
    --out-dir results/v3_beta3_1 \
    --mask-beta-a 3.0 --mask-beta-b 1.0 \
    $COMMON

echo ">>> Phase 1, Run 3: Beta(5,1)"
uv run python scripts/train_diffusion_model.py \
    --out-dir results/v3_beta5_1 \
    --mask-beta-a 5.0 --mask-beta-b 1.0 \
    $COMMON

# ===== Phase 1 inference: run on best checkpoints =====
echo ">>> Phase 1 inference sweeps"
for dir in results/v3_beta2_1 results/v3_beta3_1 results/v3_beta5_1; do
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

echo ">>> Phase 1 complete. Check gen_val in logs to pick best Beta."
echo ">>> Then uncomment Phase 2 runs below with the winning Beta values."

# ===== Phase 2: Regularization sweeps (uncomment after Phase 1) =====
# Replace BEST_A and BEST_B with the winning Beta parameters from Phase 1.

# BEST_A=3.0
# BEST_B=1.0

# echo ">>> Phase 2, Run 1: Best Beta + dropout=0.2"
# uv run python scripts/train_diffusion_model.py \
#     --out-dir results/v3_best_drop02 \
#     --mask-beta-a $BEST_A --mask-beta-b $BEST_B \
#     --dropout 0.2 \
#     $COMMON

# echo ">>> Phase 2, Run 3: Best Beta + weight_decay=0.05"
# uv run python scripts/train_diffusion_model.py \
#     --out-dir results/v3_best_wd05 \
#     --mask-beta-a $BEST_A --mask-beta-b $BEST_B \
#     --weight-decay 0.05 \
#     $COMMON

# echo ">>> Phase 2, Run 4: Best Beta + weight_decay=0.1"
# uv run python scripts/train_diffusion_model.py \
#     --out-dir results/v3_best_wd10 \
#     --mask-beta-a $BEST_A --mask-beta-b $BEST_B \
#     --weight-decay 0.1 \
#     $COMMON

echo "=== All done ==="

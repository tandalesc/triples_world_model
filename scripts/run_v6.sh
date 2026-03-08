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

# Run 1: Baseline (v2 architecture, long training, gen_val_tok tracking)
echo ">>> Run 1: v6_baseline (v2 arch, 600 epochs, gen_val_tok early stop)"
uv run python scripts/train_diffusion_v6.py \
    --out-dir results/v6_baseline \
    $COMMON

# Run 2: FiLM + cross-attention (long training)
echo ">>> Run 2: v6_film (FiLM + xattn, 600 epochs, gen_val_tok early stop)"
uv run python scripts/train_diffusion_v6.py \
    --out-dir results/v6_film \
    --use-film \
    $COMMON

# Inference on best checkpoints
echo ">>> Inference sweeps"
for dir in results/v6_baseline results/v6_film; do
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

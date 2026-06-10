#!/bin/bash
set -e
export OMP_NUM_THREADS=1
cd ~/twm_diffusion_package

# Build domain BPE tokenizer if it doesn't exist
if [ ! -f data/atomic_10000/domain_bpe_tokenizer.json ]; then
    echo ">>> Building domain BPE tokenizer..."
    uv run python scripts/build_domain_vocab.py \
        --data-dir data/atomic_10000 \
        --vocab-size 1500
fi

COMMON="--data-dir data/atomic_10000 \
    --domain-tokenizer data/atomic_10000/domain_bpe_tokenizer.json \
    --config base \
    --denoiser-layers 1 --denoiser-heads 4 \
    --max-value-tokens 12 \
    --pretrained-dynamics pretrained/model_best.pt \
    --freeze-dynamics --freeze-encoder \
    --epochs 600 --batch-size 32 --lr 3e-4 \
    --patience 100 --log-every 10 --diagnostic-every 50 \
    --use-adaln --use-continuous-noise --unified-decoder --wspace \
    --alpha-min 0.01 --timestep-bias-power 2.0 \
    --device cuda:1"

# Run 1: MSE x₀-prediction (primary experiment)
echo ">>> Run 1: v12d_mse (MSE x₀-prediction, W-space)"
uv run python scripts/train_diffusion_v12d.py \
    --out-dir results/v12d_mse \
    $COMMON

# Inference on best checkpoint
echo ">>> Inference"
if [ -f results/v12d_mse/model_best.pt ]; then
    echo "  v12d_mse: steps=10, temp=0.0"
    uv run python scripts/infer_diffusion_model.py \
        --model-dir results/v12d_mse \
        --data-dir data/atomic_10000 \
        --denoise-steps 10 \
        --temperature 0.0 \
        --split test --n-examples 30 \
        --device cuda:1 2>&1 | tee results/v12d_mse/infer_steps10_temp0.0.txt
fi

echo "=== All done ==="

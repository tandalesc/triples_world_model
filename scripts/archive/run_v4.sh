#!/bin/bash
set -e
export OMP_NUM_THREADS=1
cd ~/twm_diffusion_package

echo ">>> v4: Domain vocab, 1L/128d, uniform masking"
uv run python scripts/train_diffusion_v4.py \
    --data-dir data/atomic_10000 \
    --out-dir results/v4_domain_vocab \
    --config base \
    --denoiser-layers 1 --denoiser-dim 128 --denoiser-heads 4 \
    --n-proj-tokens 4 \
    --max-word-tokens 10 --min-word-count 3 \
    --pretrained-dynamics pretrained/model_best.pt \
    --freeze-dynamics --freeze-encoder \
    --epochs 300 --batch-size 32 --lr 3e-4 \
    --patience 50 --log-every 10 --device cuda:1

echo "=== Done ==="

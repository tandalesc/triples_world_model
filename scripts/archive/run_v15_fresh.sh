#!/bin/bash
set -e
export OMP_NUM_THREADS=1
cd ~/twm_diffusion_package

COMMON="--config base \
    --denoiser-heads 4 \
    --compressor-layers 2 \
    --max-value-tokens 12 \
    --batch-size 32 --lr 3e-4 \
    --log-every 10 --diagnostic-every 50 \
    --alpha-min 0.01 \
    --dropout 0.1 \
    --aux-ce-weight 0.1 \
    --length-weight 0.1 \
    --phase1-epochs 100 \
    --phase2-epochs 200 \
    --phase2-patience 50 \
    --phase2-bias-power 2.0 \
    --device cuda:1"

DATA="--data-dir data/atomic_10000_identity \
    --domain-tokenizer data/atomic_10000/domain_bpe_tokenizer.json"

# v15a: 1L denoiser, fresh
echo ">>> v15a: 1L denoiser, fresh, curriculum"
uv run python scripts/train_v15_fresh.py \
    $DATA \
    --out-dir results/v15a_1L_fresh \
    --denoiser-layers 1 \
    $COMMON

# v15b: 2L denoiser, fresh
echo ">>> v15b: 2L denoiser, fresh, curriculum"
uv run python scripts/train_v15_fresh.py \
    $DATA \
    --out-dir results/v15b_2L_fresh \
    --denoiser-layers 2 \
    $COMMON

echo "=== All done ==="

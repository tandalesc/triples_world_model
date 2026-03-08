#!/bin/bash
set -e
export OMP_NUM_THREADS=1
cd ~/twm_diffusion_package

COMMON="--config base \
    --denoiser-layers 1 --denoiser-heads 4 \
    --compressor-layers 2 \
    --max-value-tokens 12 \
    --batch-size 32 --lr 3e-4 \
    --log-every 10 --diagnostic-every 50 \
    --use-adaln --use-continuous-noise --unified-decoder --wspace \
    --alpha-min 0.01 \
    --use-decode-proj \
    --epochs 300 --patience 50 \
    --device cuda:1"

# ATOMIC only
echo ">>> v14 Compressor/Expander identity — ATOMIC"
uv run python scripts/train_v14_compress.py \
    --data-dir data/atomic_10000_identity \
    --out-dir results/v14_compress_atomic \
    --domain-tokenizer data/atomic_10000/domain_bpe_tokenizer.json \
    $COMMON

echo "=== All done ==="

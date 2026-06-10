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

echo ">>> v12c_curriculum: three-phase curriculum training"
uv run python scripts/train_diffusion_v12c.py \
    --data-dir data/atomic_10000 \
    --out-dir results/v12c_curriculum \
    --domain-tokenizer data/atomic_10000/domain_bpe_tokenizer.json \
    --config base \
    --denoiser-layers 1 --denoiser-heads 4 \
    --max-value-tokens 12 \
    --pretrained-dynamics pretrained/model_best.pt \
    --freeze-dynamics --freeze-encoder \
    --batch-size 32 --lr 3e-4 \
    --diagnostic-every 50 \
    --use-adaln --use-continuous-noise --unified-decoder --wspace \
    --alpha-min 0.01 --timestep-bias-power 2.0 \
    --phase1-max-epochs 200 \
    --phase2-max-epochs 200 \
    --phase3-max-epochs 200 \
    --device cuda:1

# Inference on best checkpoint
echo ">>> Inference"
if [ -f results/v12c_curriculum/model_best.pt ]; then
    echo "  Best overall checkpoint:"
    uv run python scripts/infer_diffusion_model.py \
        --model-dir results/v12c_curriculum \
        --data-dir data/atomic_10000 \
        --denoise-steps 10 \
        --temperature 0.0 \
        --split test --n-examples 30 \
        --device cuda:1 2>&1 | tee results/v12c_curriculum/infer_best.txt
fi

echo "=== All done ==="

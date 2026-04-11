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

# Run 1: Structured noise + loss reweighting (primary experiment)
echo ">>> Run 1: v12b_structured_noise (structured noise + reweight)"
uv run python scripts/train_diffusion_v12b.py \
    --out-dir results/v12b_structured_noise \
    --structured-noise --noise-neighbors 10 \
    --timestep-weight-scale 2.0 \
    $COMMON

# Run 2: Reweight only (isotropic noise + loss reweighting)
echo ">>> Run 2: v12b_reweight_only (isotropic noise + reweight)"
uv run python scripts/train_diffusion_v12b.py \
    --out-dir results/v12b_reweight_only \
    --timestep-weight-scale 2.0 \
    $COMMON

# Run 3: Structured noise only (no loss reweighting)
echo ">>> Run 3: v12b_structured_only (structured noise, no reweight)"
uv run python scripts/train_diffusion_v12b.py \
    --out-dir results/v12b_structured_only \
    --structured-noise --noise-neighbors 10 \
    $COMMON

# Inference on best checkpoints
echo ">>> Inference"
for dir in results/v12b_structured_noise results/v12b_reweight_only results/v12b_structured_only; do
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

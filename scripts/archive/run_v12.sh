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

# Run 1: W-space decoding, embeddings initialized from TWM encoder
echo ">>> Run 1: v12_wspace (W-space, encoder-init embeddings)"
uv run python scripts/train_diffusion_v12.py \
    --out-dir results/v12_wspace \
    $COMMON

# Run 2: W-space decoding, random embedding init (ablation)
echo ">>> Run 2: v12_wspace_randinit (W-space, random embeddings)"
uv run python scripts/train_diffusion_v12.py \
    --out-dir results/v12_wspace_randinit \
    --no-wspace-init \
    $COMMON

# Inference on best checkpoints
echo ">>> Inference"
for dir in results/v12_wspace results/v12_wspace_randinit; do
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

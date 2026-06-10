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
    --cond-drop-prob 0.15 \
    --device cuda:1"

# Run 1: t=1.0 fixed, MSE loss (diagnostic — can the model construct at all?)
echo ">>> Run 1: v12f_t1_mse (MSE, fixed t=1.0, W-space)"
uv run python scripts/train_diffusion_v12f.py \
    --out-dir results/v12f_t1_mse \
    --fixed-timestep 1.0 --loss-type mse \
    $COMMON

# Run 2: Full range, MSE + CFG (main experiment)
echo ">>> Run 2: v12f_mse_cfg (MSE + CFG, W-space)"
uv run python scripts/train_diffusion_v12f.py \
    --out-dir results/v12f_mse_cfg \
    --loss-type mse \
    $COMMON

# Inference sweep on best checkpoints
echo ">>> Inference sweep"
for DIR in results/v12f_t1_mse results/v12f_mse_cfg; do
    if [ -f $DIR/model_best.pt ]; then
        for GS in 1.0 1.5 2.0 3.0 5.0 7.0 10.0; do
            echo "  $(basename $DIR): steps=10, guidance_scale=$GS"
            uv run python scripts/infer_diffusion_model.py \
                --model-dir $DIR \
                --data-dir data/atomic_10000 \
                --denoise-steps 10 \
                --temperature 0.0 \
                --guidance-scale $GS \
                --split test --n-examples 30 \
                --device cuda:1 2>&1 | tee ${DIR}/infer_gs${GS}.txt
        done
    fi
done

echo "=== All done ==="

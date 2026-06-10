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
    --batch-size 32 --lr 3e-4 \
    --log-every 10 --diagnostic-every 50 \
    --use-adaln --use-continuous-noise --unified-decoder --wspace \
    --alpha-min 0.01 --timestep-bias-power 2.0 \
    --cond-drop-prob 0.15 \
    --device cuda:1"

# Find phase 1 checkpoint
P1_CKPT="results/v12f_t1_mse/model_best.pt"
if [ ! -f "$P1_CKPT" ]; then
    echo "ERROR: Phase 1 checkpoint not found at $P1_CKPT"
    echo "Run v12f first: bash scripts/run_v12f.sh"
    exit 1
fi

# Run 1: CE output head — W-space denoiser with learned readout
echo ">>> v12g_ce: curriculum phases 2-4, CE output head"
uv run python scripts/train_diffusion_v12g.py \
    --out-dir results/v12g_ce \
    --phase1-checkpoint $P1_CKPT \
    --phase2-epochs 200 --phase2-patience 50 \
    --phase3-epochs 200 --phase3-patience 50 \
    --phase4-epochs 200 --phase4-patience 100 \
    --use-output-head --no-decode-proj \
    $COMMON

# Run 2: MSE + no proj (baseline comparison)
echo ">>> v12g_mse: curriculum phases 2-4, MSE only"
uv run python scripts/train_diffusion_v12g.py \
    --out-dir results/v12g_mse \
    --phase1-checkpoint $P1_CKPT \
    --phase2-epochs 200 --phase2-patience 50 \
    --phase3-epochs 200 --phase3-patience 50 \
    --phase4-epochs 200 --phase4-patience 100 \
    --no-decode-proj \
    $COMMON

# Inference sweep on best checkpoints
echo ">>> Inference sweep"
for DIR in results/v12g_ce results/v12g_mse; do
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

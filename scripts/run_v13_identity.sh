#!/bin/bash
set -e
export OMP_NUM_THREADS=1
cd ~/twm_diffusion_package

# Step 1: Prepare WebNLG data
if [ ! -f data/webnlg/train.jsonl ]; then
    echo ">>> Preparing WebNLG data..."
    uv run python scripts/prepare_webnlg.py --out-dir data/webnlg
fi

# Step 2: Build domain BPE tokenizer on WebNLG vocab
if [ ! -f data/webnlg/domain_bpe_tokenizer.json ]; then
    echo ">>> Building domain BPE tokenizer for WebNLG..."
    uv run python scripts/build_domain_vocab.py \
        --data-dir data/webnlg \
        --vocab-size 1500
fi

COMMON="--data-dir data/webnlg \
    --domain-tokenizer data/webnlg/domain_bpe_tokenizer.json \
    --config base \
    --denoiser-layers 1 --denoiser-heads 4 \
    --max-value-tokens 12 \
    --batch-size 32 --lr 3e-4 \
    --log-every 10 --diagnostic-every 50 \
    --use-adaln --use-continuous-noise --unified-decoder --wspace \
    --alpha-min 0.01 --timestep-bias-power 2.0 \
    --cond-drop-prob 0.15 \
    --device cuda:1"

# Run 1: MSE + decode_proj (same setup that worked in v12f, now with identity advance)
echo ">>> v13_identity_mse: MSE + decode_proj, identity advance on WebNLG"
uv run python scripts/train_v13_identity.py \
    --out-dir results/v13_identity_mse \
    --use-decode-proj \
    --phase1-epochs 200 --phase1-patience 50 \
    --phase2-epochs 200 --phase2-patience 50 \
    --phase3-epochs 200 --phase3-patience 100 \
    $COMMON

# Run 2: CE output head (if MSE works, try CE for comparison)
echo ">>> v13_identity_ce: CE output head, identity advance on WebNLG"
uv run python scripts/train_v13_identity.py \
    --out-dir results/v13_identity_ce \
    --use-output-head \
    --phase1-epochs 200 --phase1-patience 50 \
    --phase2-epochs 200 --phase2-patience 50 \
    --phase3-epochs 200 --phase3-patience 100 \
    $COMMON

# Inference sweep on best checkpoints
echo ">>> Inference sweep"
for DIR in results/v13_identity_mse results/v13_identity_ce; do
    if [ -f $DIR/model_best.pt ]; then
        for GS in 1.0 3.0 5.0 7.0; do
            echo "  $(basename $DIR): steps=10, guidance_scale=$GS"
            uv run python scripts/infer_diffusion_model.py \
                --model-dir $DIR \
                --data-dir data/webnlg \
                --denoise-steps 10 \
                --temperature 0.0 \
                --guidance-scale $GS \
                --split test --n-examples 30 \
                --device cuda:1 2>&1 | tee ${DIR}/infer_gs${GS}.txt
        done
    fi
done

echo "=== All done ==="

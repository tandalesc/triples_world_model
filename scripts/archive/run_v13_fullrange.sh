#!/bin/bash
set -e
export OMP_NUM_THREADS=1
cd ~/twm_diffusion_package

# Step 1: Run check_exact_match on both existing runs
echo ">>> Exact match diagnostic — WebNLG"
WEBNLG_DIR=results/v13_identity_webnlg
if [ -d "$WEBNLG_DIR/phase1" ] && [ -f "$WEBNLG_DIR/phase1/model_best.pt" ]; then
    uv run python scripts/check_exact_match.py \
        --model-dir $WEBNLG_DIR/phase1 \
        --data-dir data/webnlg \
        --n-examples 5 \
        --device cuda:1 2>&1 | tee $WEBNLG_DIR/exact_match_diag.txt
fi

echo ""
echo ">>> Exact match diagnostic — ATOMIC"
ATOMIC_DIR=results/v13_identity_atomic
if [ -d "$ATOMIC_DIR/phase1" ] && [ -f "$ATOMIC_DIR/phase1/model_best.pt" ]; then
    uv run python scripts/check_exact_match.py \
        --model-dir $ATOMIC_DIR/phase1 \
        --data-dir data/atomic_10000_identity \
        --n-examples 5 \
        --device cuda:1 2>&1 | tee $ATOMIC_DIR/exact_match_diag.txt
fi

# Step 2: Full range [0.0, 1.0] — WebNLG
# Find best checkpoint from whatever phase completed
WEBNLG_CKPT=""
for PHASE in phase3 phase2 phase1; do
    if [ -f "$WEBNLG_DIR/$PHASE/model_best.pt" ]; then
        WEBNLG_CKPT="$WEBNLG_DIR/$PHASE/model_best.pt"
        echo ">>> WebNLG: using checkpoint from $PHASE"
        break
    fi
done

if [ -n "$WEBNLG_CKPT" ]; then
    echo ">>> v13 WebNLG full range [0.0, 1.0]"
    uv run python scripts/train_v13_identity.py \
        --data-dir data/webnlg \
        --out-dir results/v13_webnlg_fullrange \
        --domain-tokenizer data/webnlg/domain_bpe_tokenizer.json \
        --config base \
        --denoiser-layers 1 --denoiser-heads 4 \
        --max-value-tokens 12 \
        --batch-size 32 --lr 3e-4 \
        --log-every 10 --diagnostic-every 50 \
        --use-adaln --use-continuous-noise --unified-decoder --wspace \
        --alpha-min 0.01 --timestep-bias-power 2.0 \
        --cond-drop-prob 0.15 \
        --use-decode-proj \
        --resume-checkpoint "$WEBNLG_CKPT" \
        --phase1-epochs 0 --phase1-patience 1 \
        --phase2-epochs 0 --phase2-patience 1 \
        --phase3-epochs 200 --phase3-patience 50 \
        --device cuda:1
fi

# Step 3: Full range [0.0, 1.0] — ATOMIC
ATOMIC_CKPT=""
for PHASE in phase3 phase2 phase1; do
    if [ -f "$ATOMIC_DIR/$PHASE/model_best.pt" ]; then
        ATOMIC_CKPT="$ATOMIC_DIR/$PHASE/model_best.pt"
        echo ">>> ATOMIC: using checkpoint from $PHASE"
        break
    fi
done

if [ -n "$ATOMIC_CKPT" ]; then
    echo ">>> v13 ATOMIC full range [0.0, 1.0]"
    uv run python scripts/train_v13_identity.py \
        --data-dir data/atomic_10000_identity \
        --out-dir results/v13_atomic_fullrange \
        --domain-tokenizer data/atomic_10000/domain_bpe_tokenizer.json \
        --config base \
        --denoiser-layers 1 --denoiser-heads 4 \
        --max-value-tokens 12 \
        --batch-size 32 --lr 3e-4 \
        --log-every 10 --diagnostic-every 50 \
        --use-adaln --use-continuous-noise --unified-decoder --wspace \
        --alpha-min 0.01 --timestep-bias-power 2.0 \
        --cond-drop-prob 0.15 \
        --use-decode-proj \
        --resume-checkpoint "$ATOMIC_CKPT" \
        --phase1-epochs 0 --phase1-patience 1 \
        --phase2-epochs 0 --phase2-patience 1 \
        --phase3-epochs 200 --phase3-patience 50 \
        --device cuda:1
fi

echo "=== Full range training complete ==="

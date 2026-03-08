#!/bin/bash
set -e
export OMP_NUM_THREADS=1
cd ~/twm_diffusion_package

echo ">>> Exact match check — WebNLG"
WEBNLG_DIR=results/v13_identity_webnlg
WEBNLG_CKPT=""
for PHASE in phase3 phase2 phase1; do
    if [ -f "$WEBNLG_DIR/$PHASE/model_best.pt" ]; then
        WEBNLG_CKPT="$WEBNLG_DIR/$PHASE"
        echo "Using $PHASE checkpoint"
        break
    fi
done

if [ -n "$WEBNLG_CKPT" ]; then
    uv run python scripts/check_exact_match.py \
        --model-dir "$WEBNLG_CKPT" \
        --data-dir data/webnlg \
        --n-examples 30 \
        --denoise-steps 10 \
        --device cuda:1 2>&1 | tee $WEBNLG_DIR/exact_match_check.txt
else
    echo "No WebNLG checkpoint found"
fi

echo ""
echo ">>> Exact match check — ATOMIC"
ATOMIC_DIR=results/v13_identity_atomic
ATOMIC_CKPT=""
for PHASE in phase3 phase2 phase1; do
    if [ -f "$ATOMIC_DIR/$PHASE/model_best.pt" ]; then
        ATOMIC_CKPT="$ATOMIC_DIR/$PHASE"
        echo "Using $PHASE checkpoint"
        break
    fi
done

if [ -n "$ATOMIC_CKPT" ]; then
    uv run python scripts/check_exact_match.py \
        --model-dir "$ATOMIC_CKPT" \
        --data-dir data/atomic_10000_identity \
        --n-examples 30 \
        --denoise-steps 10 \
        --device cuda:1 2>&1 | tee $ATOMIC_DIR/exact_match_check.txt
else
    echo "No ATOMIC checkpoint found"
fi

echo ""
echo "=== Done ==="

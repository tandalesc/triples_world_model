#!/bin/zsh
# Deep Sets MLP baseline (each triple read on its own before summing): the runs
# behind the paper's architecture comparison and withheld-pair test, with the
# settings of the original MLP runs.
cd "$(dirname "$0")/../.."
D=/Users/shishirtandale/Projects/hearthvale_decisions/.claude/worktrees/decision-distillation-review-bfaebb/data/distill
R=results/decision_student
run() { local out=$1; shift; echo "[$(date +%T)] $out"; .venv/bin/python scripts/train_decision_student.py "$@" --arch mlp --out $out > $out.log 2>&1 || echo "  failed: $(tail -2 $out.log)"; grep -E "^all|test KL" $out.log | tail -1; }
run $R/cmp-ds/mlp-h128-p1 $D/train-a.jsonl --epochs 20 --batch-size 1024 --lr 0.003 --weight-decay 0.05 --d-model 32 --mlp-hidden 128 --dropout 0.1
run $R/cmp-ds/mlp-h128-p3 $D/train-a.jsonl --epochs 20 --batch-size 1024 --lr 0.003 --weight-decay 0.05 --d-model 32 --mlp-hidden 128 --dropout 0.3
run $R/cmp-ds/mlp-h256-p3 $D/train-a.jsonl --epochs 20 --batch-size 1024 --lr 0.003 --weight-decay 0.05 --d-model 32 --mlp-hidden 256 --dropout 0.3
run $R/cmp-ds/mlp-d64-h256-p3 $D/train-a.jsonl --epochs 20 --batch-size 1024 --lr 0.003 --weight-decay 0.05 --d-model 64 --mlp-hidden 256 --dropout 0.3
run $R/cmp-ds/aux-mlp-d64-h256 $D/train-a.jsonl $D/relabelled-set.jsonl --epochs 12 --batch-size 2048 --lr 0.003 --weight-decay 0.05 --d-model 64 --mlp-hidden 256 --dropout 0.3
run $R/holdout-ds/mlp-d64-h256 $D/train-a.jsonl --holdout-pairs configs/decision_student/holdout_pairs.json --epochs 20 --batch-size 1024 --lr 0.003 --weight-decay 0.05 --d-model 64 --mlp-hidden 256 --dropout 0.3
echo "[$(date +%T)] done"

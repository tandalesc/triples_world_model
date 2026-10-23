#!/bin/zsh
# Attention (set model) vs Deep Sets MLP, trained from scratch on the same data,
# then played on the teacher reference's towns (4 towns x 2 runs, 150 years,
# simulated player) on the current simulation.
set -e
TWM=/Users/shishirtandale/Projects/twm-decision-student
HV=/Users/shishirtandale/Projects/hearthvale_decisions/.claude/worktrees/decision-distillation-review-bfaebb
SIM=/Users/shishirtandale/Projects/hearthvale_decisions/.claude/worktrees/deploy-merge
D=$HV/data/distill
DATA=($D/train-a.jsonl $D/onpolicy-1.jsonl $HV/data/loop/v2/novel-rows.jsonl $D/young-v4/rows.jsonl)
R=$TWM/results/decision_student/arch-towns
mkdir -p $R
cd $TWM
echo "[$(date +%T)] training set model"
.venv/bin/python scripts/train_decision_student.py $DATA --arch set --epochs 20 --lr 0.003 --d-model 32 --d-ff 64 --heads 2 --state-layers 0 --question-layers 1 --min-value-count 3 --seed 0 --out $R/set > $R/set.log 2>&1
grep -E "^all" $R/set.log | tail -1
echo "[$(date +%T)] training Deep Sets MLP"
.venv/bin/python scripts/train_decision_student.py $DATA --arch mlp --epochs 20 --batch-size 1024 --lr 0.003 --weight-decay 0.05 --d-model 64 --mlp-hidden 256 --dropout 0.3 --min-value-count 3 --seed 0 --out $R/mlp > $R/mlp.log 2>&1
grep -E "^all" $R/mlp.log | tail -1
for m in set mlp; do .venv/bin/python scripts/export_decision_student.py $R/$m $R/$m.bundle.json --data $D/train-a.jsonl 2>&1 | tail -1; done
cd $SIM
for m in set mlp; do
  for salt in 0 1; do
    echo "[$(date +%T)] towns: $m salt $salt"
    node --max-old-space-size=8000 scripts/distill-collect.js $R/towns-$m-$salt --seeds 4 --start 65 --salt-start $salt --salts 1 --years 150 --workers 4 \
      --answer student --student $R/$m.bundle.json --player --no-record > $R/towns-$m-$salt.log 2>&1
    grep -E "^seed" $R/towns-$m-$salt.log
  done
done
echo "[$(date +%T)] done"

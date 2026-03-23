# Kaggle Titanic Experiments with Triple World Model

## Approach

We frame tabular classification as a **state transition** over structured triples. Instead of a traditional classifier, we use a small transformer that operates on `(entity, attribute, value)` triples.

Given a passenger's attributes as input triples, the model predicts the same attributes plus the target label (Survived / Transported) as output triples. The transformer learns which attribute combinations predict the outcome through its attention mechanism — no feature engineering beyond discretization.

### Example (Classic Titanic)

**Input (state_t):**
```
[#mode, type, advance]
[passenger, class, third]
[passenger, sex, male]
[passenger, age_group, young_adult]
[passenger, family_size, solo]
[passenger, fare, low]
[passenger, embarked, southampton]
[passenger, cabin_known, no]
```

**Output (state_t+1):**
```
[passenger, class, third]
[passenger, sex, male]
[passenger, age_group, young_adult]
[passenger, family_size, solo]
[passenger, fare, low]
[passenger, embarked, southampton]
[passenger, cabin_known, no]
[passenger, survived, no]
```

The model learns to reproduce the input attributes (identity) and predict the target attribute (classification) in a single forward pass.

## Models

We use the closed-vocab `TripleWorldModel` — a vanilla transformer with learned embeddings over a small fixed token set. No pre-training, no external data.

| Profile | d_model | Layers | Heads | d_ff | Params |
|---------|--------:|-------:|------:|-----:|-------:|
| micro | 16 | 1 | 2 | 32 | ~5K |
| mini | 32 | 2 | 2 | 128 | ~29K |

## Classic Titanic (891 passengers)

**Attributes (7):** class, sex, age_group, family_size, fare, embarked, cabin_known

Discretization: Age into 4 bins (child/young_adult/adult/senior), Fare into 4 bins (low/medium/high/premium), SibSp+Parch into 3 bins (solo/small/large).

Missing values handled by omitting the corresponding triple — the model learns to predict with whatever attributes are available.

### Results

| Model | Train Acc | Seen Acc | Comp Acc |
|-------|----------:|---------:|---------:|
| micro | 86.5% | **83.1%** | 54.1% |
| mini | 87.9% | 86.6% | 71.1% |

- **Seen**: random 10% holdout (same attribute combos as training) — comparable to Kaggle leaderboard
- **Comp**: passengers with attribute combinations never seen during training (compositional generalization test)
- Kaggle legitimate ML baseline (Random Forest): ~79.4%

Micro generalizes better on comp despite lower capacity — classic small-data regime where larger models overfit.

## Spaceship Titanic (8693 passengers)

**Attributes (13):** home_planet, cryo_sleep, destination, age_group, room_service, food_court, shopping, spa, vrdeck, deck, side, cabin_region, group_size

Key differences from v1: individual spending columns (not summed), cabin number binned into forward/midship/aft regions, finer age bins (added teen), partial triples for missing data (0 rows dropped vs 942 previously).

### Results (mini, v1 features)

| Split | Transported Acc |
|-------|----------------:|
| Train | 79.4% |
| Seen | 73.0% |
| Comp | 86.0% |

- Kaggle leaderboard top: ~82.5%
- v2 features (individual spending, cabin_region, partial triples) pending training

## Test Splits Explained

- **Train**: accuracy on training data
- **Seen**: held-out passengers whose attribute combos appear in training (standard generalization)
- **Comp**: held-out passengers with attribute combos never seen in training (compositional generalization — tests whether the model learned rules vs memorized patterns)

## Reproducing

```bash
# Convert data
uv run python scripts/convert_titanic.py
uv run python scripts/convert_spaceship_titanic.py

# Train
uv run python -m twm.train --data-dir data/titanic --out-dir results/titanic \
  --config micro --max-triples 12 --epochs 300 --batch-size 16 --lr 1e-3 \
  --log-every 10 --target-attr survived

uv run python -m twm.train --data-dir data/spaceship-titanic --out-dir results/spaceship_titanic \
  --config mini --max-triples 16 --epochs 300 --batch-size 32 --lr 1e-3 \
  --log-every 10 --target-attr transported

# Generate Kaggle submissions
PYTHONPATH=scripts:$PYTHONPATH uv run python scripts/predict_titanic.py \
  --checkpoint results/titanic
PYTHONPATH=scripts:$PYTHONPATH uv run python scripts/predict_spaceship_titanic.py \
  --checkpoint results/spaceship_titanic
```

## What This Demonstrates

1. **Tabular classification works as state transition.** A transformer over decomposed triples can learn classification without any task-specific architecture.
2. **Tiny models compete.** A 5-29K parameter transformer matches Random Forest baselines on Titanic with no hyperparameter tuning.
3. **Partial inputs are natural.** Missing values are handled by simply omitting triples — no imputation needed. The model learns to predict from variable-length attribute sets.
4. **Compositional generalization is measurable.** The comp split tests whether the model learned transferable rules or just memorized training patterns.

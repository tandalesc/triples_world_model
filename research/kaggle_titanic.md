# Kaggle Tabular Classification with Triple World Model

## Approach

We frame tabular classification as a **state transition** over structured triples. Instead of a traditional classifier, we use a small transformer that operates on `(entity, attribute, value)` triples.

Given a passenger's attributes as input triples, the model predicts the same attributes plus the target label as output triples. The transformer learns which attribute combinations predict the outcome through its attention mechanism — no feature engineering beyond discretization.

### Example (Classic Titanic)

**Input (state_t):**
```
[#mode, type, advance]
[passenger, title, mr]
[passenger, class, third]
[passenger, sex, male]
[passenger, age_group, young_adult]
[passenger, is_alone, yes]
[passenger, fare_pp, very_low]
[passenger, embarked, southampton]
[passenger, cabin_known, no]
```

**Output (state_t+1):**
```
[passenger, title, mr]
[passenger, class, third]
[passenger, sex, male]
[passenger, age_group, young_adult]
[passenger, is_alone, yes]
[passenger, fare_pp, very_low]
[passenger, embarked, southampton]
[passenger, cabin_known, no]
[passenger, survived, no]
```

## Models

Closed-vocab `TripleWorldModel` — a vanilla transformer with learned embeddings over a small fixed token set. No pre-training, no external data.

| Profile | d_model | Layers | Heads | d_ff | Params |
|---------|--------:|-------:|------:|-----:|-------:|
| micro   |      16 |      1 |     2 |   32 |   ~4K  |
| mini    |      32 |      2 |     2 |  128 |  ~30K  |

## Classic Titanic — Kaggle Scores

891 training passengers, 418 test passengers. Target: Survived (yes/no).

### Feature Evolution

- **v1** (7 attrs): class, sex, age_group, family_size, fare, embarked, cabin_known
- **v2** (13 attrs): added title (mr/mrs/miss/master/rare), finer age bins (infant/child/teen), sibsp/parch separately, is_alone, is_child, fare per person, cabin deck, age_estimated

Title extraction from Name was the single most impactful feature — Mr survives at 16%, Mrs at 79%, Miss at 70%, Master at 57%.

### Kaggle Submission Results

| Version | Model | Params | Kaggle Score |
|---------|-------|-------:|-------------:|
| v1      | micro |   ~4K  |        0.741 |
| v1      | mini  |  ~31K  |        0.763 |
| v2      | micro |   ~4K  |        **0.768** |
| v2      | mini  |  ~31K  |        0.763 |

**Best: micro v2 at 76.8%** — competitive with Random Forest baselines (~79.4%). Micro outperforms mini on v2 features, consistent with small-data regime where less capacity = less overfitting.

### Key Findings

- **Title is essential.** Adding title extraction improved micro from 74.1% to 76.8%.
- **Micro > mini for 891 examples.** Less capacity forces the model to learn generalizable rules.
- **Internal test splits are misleading.** Our "seen" accuracy was 83% but Kaggle score was 74%. The only honest eval is submission.
- **Identity examples don't help classification.** They dilute training signal — every example should teach the prediction task.

## Spaceship Titanic — Kaggle Scores

8693 training passengers, 4277 test passengers. Target: Transported (true/false).

### Feature Engineering

13 attributes: home_planet, cryo_sleep, destination, age_group, room_service, food_court, shopping, spa, vrdeck, deck, side, cabin_region, group_size.

Key data insights:
- **CryoSleep is the strongest predictor** — 82% of cryo passengers are transported vs 33% non-cryo.
- **Spending signal is mostly zero vs nonzero** — once you spend anything, transported rate drops from ~60% to ~28%. Amount beyond zero matters less.
- **Spending direction differs by column** — RoomService/Spa/VRDeck: more spending = less transported. FoodCourt/ShoppingMall: more spending = slightly more transported.
- **Cabin region matters** — forward/midship/aft correlates with deck and transported rate.

### Kaggle Submission Results

| Version | Changes | Kaggle Score |
|---------|---------|-------------:|
| v3 (baseline) | 3-bin spending (none/low/high), identity examples, internal splits | 0.779 |
| v4 | 4-bin spending, 500 epochs | 0.775 |
| v5 | zero/nonzero spending, no identity, no splits | 0.753 |
| v6 | v3 baseline + dropout 0.2 | **0.787** |

**Best: v6 at 78.7%** (mini, 30K params). Kaggle leaderboard top is ~82.5%.

### Key Findings

- **Identity examples help here.** Unlike Titanic (891 rows), with 8693 rows identity acts as regularization. Removing it hurt (-2.6 points).
- **Dropout 0.2 > 0.1.** The best score came from increasing dropout, confirming overfitting is the main bottleneck.
- **Simpler bins aren't always better.** Zero/nonzero spending lost information that the 3-bin scheme captured. Data analysis showed zero/nonzero is the clearest signal, but the model benefited from finer granularity.
- **More epochs can hurt.** v4 trained for 500 epochs vs 200 and scored worse — the model memorized rather than generalized.

## Personality (Playground Series S5E7) — Kaggle Scores

18,524 training examples. Target: Personality (Extrovert/Introvert).

7 attributes: time_alone, stage_fear, social_events, going_outside, drained_social, friends, post_freq.

| Model | Params | Public Score | Private Score |
|-------|-------:|-------------:|--------------:|
| mini  |  ~30K  |        0.974 |         0.969 |

**97.4% public / 96.9% private** — near-perfect classification. Clean dataset with features that directly describe the target behavior. Demonstrates TWM handles straightforward tabular tasks easily when given enough data and clear signal.

## Reproducing

```bash
# Convert data
uv run python scripts/convert_titanic.py
uv run python scripts/convert_spaceship_titanic.py
uv run python scripts/convert_personality.py

# Train (best configs)
# Titanic — micro, v2 features
uv run python -m twm.train --data-dir data/titanic --out-dir results/titanic \
  --config micro --max-triples 16 --epochs 300 --batch-size 16 --lr 1e-3 \
  --log-every 10 --target-attr survived

# Spaceship — mini, dropout 0.2
uv run python -m twm.train --data-dir data/spaceship-titanic --out-dir results/spaceship_titanic \
  --config mini --max-triples 16 --epochs 200 --batch-size 32 --lr 1e-3 \
  --log-every 10 --target-attr transported --dropout 0.2

# Personality — mini
uv run python -m twm.train --data-dir data/playground-series-s5e7 --out-dir results/personality \
  --config mini --max-triples 8 --epochs 200 --batch-size 64 --lr 1e-3 \
  --log-every 10 --target-attr personality

# Generate Kaggle submissions
PYTHONPATH=scripts:$PYTHONPATH uv run python scripts/predict_titanic.py \
  --checkpoint results/titanic
PYTHONPATH=scripts:$PYTHONPATH uv run python scripts/predict_spaceship_titanic.py \
  --checkpoint results/spaceship_titanic
PYTHONPATH=scripts:$PYTHONPATH uv run python scripts/predict_personality.py \
  --checkpoint results/personality
```

## What This Demonstrates

1. **Tabular classification works as state transition.** A transformer over decomposed triples can learn classification without any task-specific architecture.
2. **Tiny models compete.** A 4-30K parameter transformer reaches 76.8% on Titanic (vs 79.4% Random Forest), 78.7% on Spaceship Titanic (vs 82.5% top), and 97.4% on Personality.
3. **Partial inputs are natural.** Missing values are handled by simply omitting triples — no imputation needed.
4. **Data analysis drives performance.** The biggest gains came from understanding the data (title extraction, spending distributions) not from model changes.
5. **Regularization matters more than capacity.** Dropout, identity examples, and fewer epochs all helped more than increasing model size.

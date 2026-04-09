# NBA Betting Prediction Model — Team Report
**Date:** April 2026  
**Model Version:** XGBoost v1 (Post Pipeline Fix)  
**Status:** Baseline Established — Improvement Phase Starting

---

## Context: Where We Are

We have completed the first full training pipeline for the NBA game outcome prediction model.
All known data leakage issues in the pipeline have been resolved. The numbers below
represent **clean, trustworthy baseline results** — what the model genuinely knows
from historical data with no future information bleeding in.

This is our starting point. Everything from here is about closing the gap to 70%.

---

## Current Model Performance (Season 2024 Test Set)

| Metric | Value | Meaning |
|---|---|---|
| **Test Accuracy** | **64.3%** | Model correctly predicts game winner 64.3% of the time |
| **Baseline Accuracy** | 55.1% | Accuracy if we blindly predicted home team wins every game |
| **Improvement over baseline** | **+9.2pp** | Real signal the model has learned |
| **ROC-AUC** | 0.7092 | How well model separates wins from losses in probability space (0.5 = random, 1.0 = perfect) |
| **Log Loss** | 0.6141 | Quality of probability estimates (lower = better; random = 0.693) |
| **Brier Score** | 0.2152 | Mean squared error of probabilities (lower = better; random = 0.25) |
| **Best CV AUC** | 0.7157 | Cross-validation AUC on training data — only 0.007 gap vs test, model is NOT overfitting |

**Test set:** 879 games from the 2024 NBA season (never seen during training)  
**Training data:** 25,929 games across 18 seasons (2003–2021)  
**Features used:** 69 engineered features

---

## What These Numbers Actually Mean

### Accuracy: 64.3%
The model picks the correct winner in roughly **566 out of 879** test games.
For reference:
- A coin flip gives you 50%
- Always picking the home team gives 55.1%
- Our model gives 64.3%
- World-class sports models with proprietary data reach 68–72%

This is a solid, legitimate result for a model trained purely on public box-score data.

### ROC-AUC: 0.709
This tells us the model is good at *ranking* games by confidence.
When it says "I'm 70% sure the home team wins," it's right more often
than when it says "I'm 55% sure." The model's confidence is informative.
This matters for betting — we want to bet only on high-confidence games.

### CV AUC (0.7157) vs Test AUC (0.7092): Gap of 0.007
This gap is very small. It means the model generalises well to unseen data —
it learned real patterns, not just the training set.
A large gap here would indicate overfitting. We do not have that problem.

### Best Iteration: 746
The final model needed 746 boosting trees before early stopping kicked in.
This is important for tuning — the hyperparameter search only explored up to
400 trees, meaning earlier searches were likely underfitting.

---

## Where the Model Currently Struggles

### Home/Away Prediction Imbalance — The Biggest Problem

```
               Precision   Recall    F1-Score   Games
Away Win         0.62       0.49       0.55      395
Home Win         0.65       0.76       0.70      484
```

- The model correctly catches **76% of home wins** but only **49% of away wins**
- It is heavily biased toward predicting home team wins
- When it predicts an away win, it is right 62% of the time — good precision, but it rarely makes that call
- **This asymmetry is leaving money on the table.** Away upsets are undervalued.

**Root cause:** Home teams win 55% of games historically, so the model learned
to default toward home wins. It needs to be taught to be more aggressive on away wins.

---

## Gap Analysis: 64.3% → 70%

We need **+5.7 percentage points** to reach 70%.
Below is where we believe those points are hiding, ranked by expected impact.

```
Current accuracy   :  64.3%
Low-hanging fixes  :  +1.5pp   (threshold tuning + class balance)
Feature additions  :  +2.5pp   (lineup data + matchup features)
Model improvements :  +1.5pp   (ensemble + better hyperparameter search)
────────────────────────────────
Projected ceiling  :  ~70%
```

---

## Path to 70% — Action Plan

### Priority 1 — Fix Away Win Bias (Expected: +1 to +1.5pp)
**No retraining needed. Can be tested immediately.**

**A. Threshold tuning**
- Currently the model predicts home win if `P(home win) > 0.50`
- Lower this threshold to `0.45` — makes the model more willing to call away wins
- Run a threshold sweep from 0.40 to 0.55 and pick optimal on validation set

**B. Class balancing via `scale_pos_weight`**
- Add `scale_pos_weight = (number of away wins) / (number of home wins)` in XGBoost
- This tells the model to weight away wins more during training
- Currently the imbalance (395 away vs 484 home wins in test set) is tilting predictions

---

### Priority 2 — Better Lineup / Injury Features (Expected: +2 to +3pp)
**Requires new data pipeline work. Highest ceiling item.**

**A. Star player availability**
- `injured_count` in current features is too coarse — it counts all injured players equally
- A star player (top 5 on roster by minutes/points) being out is worth ~3-5 points of spread
- Need a `star_player_out_home` and `star_player_out_away` binary flag

**B. Confirmed starting lineup**
- Pre-game lineup confirmations (available on NBA.com before tip-off)
- Lineup stability: teams with their regular starters perform more predictably
- This is the single highest-value data source for game-day prediction

**C. Fatigue refinement**
- Current `fatigue_load_index` exists but is basic
- Add: distance travelled in last 72 hours, time zone changes, back-to-back road games
- Teams on long road trips with cross-timezone travel perform measurably worse

---

### Priority 3 — Expand Hyperparameter Search (Expected: +0.5 to +1pp)
**Retraining required. Low effort, clean win.**

Current search found best iteration = 746 but only searched `n_estimators` up to 1000.
The model may be underfitting in the search phase.

Changes needed:
- Increase `n_estimators` ceiling to **2000** in the search grid
- Focus `learning_rate` range on `[0.005, 0.01, 0.02]` (slow learners with more trees consistently outperform)
- Increase search iterations from 50 to **100**
- Consider Bayesian optimisation (Optuna) instead of random search for better coverage

---

### Priority 4 — Add Situational / Contextual Features (Expected: +0.5 to +1pp)

These features are available in public schedule data and add meaningful signal:

| Feature | Why it matters |
|---|---|
| Playoff seeding pressure (late season) | Teams fighting for seeding play harder in March/April |
| Back-to-back game flag | Teams on zero rest lose more than the spread implies |
| Opponent-adjusted offensive rating | Raw offensive rating doesn't account for quality of defense faced |
| Rolling last-3-game momentum | Short-term form matters more than season-long averages |
| Home stand length | Teams early in long home stands perform differently than end of stand |

---

### Priority 5 — Ensemble / Model Diversity (Expected: +0.5 to +1pp)

Single models have a performance ceiling. Combining diverse models consistently
outperforms any individual model.

Recommended ensemble:
1. **XGBoost** (current) — strong on structured tabular features
2. **LightGBM** — faster, handles categorical features differently, provides diversity
3. **Logistic Regression** on top SHAP features — captures linear relationships XGBoost misses

Blend predictions: `final_prob = 0.5 * xgb + 0.3 * lgbm + 0.2 * lr`
Weights to be optimised on validation set.

---

### Priority 6 — Data Recency Weighting (Expected: +0.3 to +0.5pp)

The model trains on 2003–2021 data equally. The 2003–2012 NBA was fundamentally
different from today (fewer 3-pointers, different pace, different defensive rules).
Old games may be adding noise rather than signal.

Fix: Apply exponential decay weights — recent seasons get full weight, older seasons get downweighted.
```
weight = exp(decay_rate * (season - 2003))
```
Test decay rates: 0.05, 0.1, 0.2 — optimise on validation AUC.

---

## Realistic Accuracy Ceiling Assessment

| Scenario | Expected Accuracy |
|---|---|
| Current (public features only, fixed) | 64.3% |
| After Priority 1–3 fixes | 65–66% |
| After Priority 1–4 (full feature work) | 66–68% |
| After Priority 1–5 (ensemble) | 67–69% |
| With real-time lineup data added | **69–71%** |

**Key insight:** 70% is achievable but requires real-time lineup data (Priority 2B).
Without confirmed starting lineups, the realistic ceiling on public box-score features
alone is approximately 66–67%. The lineup data is the unlock.

---

## What NOT to Chase

The following will not meaningfully improve accuracy and risk overfitting:

- Adding more lag windows (we already have last-5, last-10 — last-20 won't help)
- More complex architectures (neural nets on tabular sports data rarely beat XGBoost)
- Tuning further without new features — we are near the ceiling for current feature set
- Betting line data as a training feature (creates leakage — the line IS the market's model)

---

## Immediate Next Steps (This Sprint)

| # | Task | Owner | Expected gain | Effort |
|---|---|---|---|---|
| 1 | Threshold sweep (0.40–0.55) on validation set | Data Scientist | +0.5–1pp | 2 hours |
| 2 | Add `scale_pos_weight` and retrain | Data Scientist | +0.5–1pp | 1 hour |
| 3 | Re-run hyperparameter search with n_estimators up to 2000 | Data Scientist | +0.3–0.5pp | 4 hours |
| 4 | Build star-player-out feature from injury reports | Data Engineer | +1–2pp | 2–3 days |
| 5 | Add back-to-back and travel distance features | Data Engineer | +0.5–1pp | 1 day |
| 6 | Implement LightGBM and blend with XGBoost | Data Scientist | +0.3–0.5pp | 1 day |

---

## Summary

The pipeline is clean. The baseline is real. At **64.3% accuracy and 0.709 AUC**,
the model has genuine predictive value beyond the home-court baseline.

The primary gap to 70% comes from three places:
1. **Model is too conservative on away wins** — fixable immediately with no retraining
2. **Missing real-time lineup data** — the highest-value single addition possible
3. **Hyperparameter search was capped too low** — easy retraining win

With disciplined execution of Priorities 1–3, we should reach **66–67%** within
the next sprint. Crossing **70% requires the lineup data pipeline** (Priority 2B).
That is the milestone to plan toward.

---

*Report prepared from training.ipynb output — Season 2024 test set evaluation.*

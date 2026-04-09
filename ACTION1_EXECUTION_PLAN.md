# Action 1 — Fix Away Win Bias: Complete Execution Plan
**Author perspective:** Senior ML Engineer (10+ years)  
**Target:** training.ipynb  
**Goal:** Fix 49% away recall without sacrificing overall model quality  
**Estimated time:** 2.5–3 hours total

---

## Code Analysis Findings (Pre-Plan)

Before writing a single line of code, here is exactly what the current notebook does
and where the problems are. Every decision in this plan is based on reading the actual code.

### Finding 1 — The bias is in the training signal, not just the threshold

Cell 17 (final model) has no `scale_pos_weight`:
```python
final_model = xgb.XGBClassifier(
    **best_params,
    eval_metric='logloss',
    early_stopping_rounds=40,
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1,
    # ← scale_pos_weight is MISSING
)
```

Training data from Cell 8:
```
Train home win rate: 0.589
→ home_wins = 15,272
→ away_wins = 10,657
→ ratio = 0.589 : 0.411 (1.43 : 1 imbalance)
```

XGBoost sees 1.43 home wins for every away win during training.
It learned: "when uncertain, predict home win — that's the safer default."
This is not a threshold problem. It is a training signal problem.
`scale_pos_weight` corrects it at the source.

### Finding 2 — Threshold of 0.5 is hardcoded in 3 places

**Cell 19** (test evaluation):
```python
test_preds = (test_probs >= 0.5).astype(int)
```

**Cell 25** (calibration accuracy):
```python
accuracy_score(y_test, (cal_probs >= 0.5).astype(int))
```

**Cell 32** (summary):
```python
summary_acc = accuracy_score(y_test, (summary_probs >= 0.5).astype(int))
```

The threshold was never optimised. It was hardcoded at 0.5 and never revisited.
Even without retraining, threshold optimisation on the validation set is free improvement.

### Finding 3 — Betting simulation already handles away bets but rarely triggers

Cell 22 has `bet_away` logic:
```python
df['bet_away'] = (
    (df['model_prob'] <= (1 - threshold)) &        # model says away wins
    (df['market_p'] - df['model_prob'] >= edge_min) # edge over market
).astype(int)
```

With 49% away recall and threshold=0.55, the model almost never outputs
`model_prob <= 0.45` confidently. After fixing bias, this path will activate
more often → more betting opportunities → potentially better ROI.

### Finding 4 — Calibration is fitted on validation set (correct)

Cell 25:
```python
calibrated_model.fit(X_valid, y_valid)  # fit calibrator on VALIDATION set
```

This is correct. After we retrain with scale_pos_weight, the calibration
must be re-run because the raw probability distribution will shift.
If we skip recalibration, `cal_probs` will be stale and misleading.

### Finding 5 — Hyperparameter search (Cell 15) does NOT need to be re-run

The search optimises `roc_auc`. ROC-AUC is a ranking metric — it is
threshold-independent and symmetric across both classes. Adding
`scale_pos_weight` does not change which parameter configuration is optimal
by AUC. The best_params found are still valid. We add `scale_pos_weight`
only to the **final model** (Cell 17), not to the search.

This saves 4–6 hours of compute.

---

## Architecture of This Fix — Two Independent Improvements

```
Action 1A — Threshold Optimisation
  ├── Cost: ZERO (no retraining)
  ├── Method: sweep thresholds on validation set, pick best, apply to test
  └── Effect: shifts decision boundary, balances recall

Action 1B — scale_pos_weight Retraining
  ├── Cost: ~10 min retraining
  ├── Method: add SPW to final model, retrain, recalibrate
  └── Effect: rebalances training signal at the source

Run 1A first (free, instant feedback).
Then 1B (retrain once with SPW).
Then 1A again on the new model (SPW shifts the optimal threshold).
```

**Critical rule:** Threshold is ALWAYS selected on the validation set.
Never on the test set. The test set is only used for final reporting.
Picking a threshold on the test set is a form of data leakage.

---

## What We Will Add / Change — Exact Cell Map

| Cell | Change type | What |
|---|---|---|
| **New cell after 17** | ADD | Threshold sweep on validation set |
| **Cell 17** | MODIFY | Add `scale_pos_weight` |
| **New cell after 17** | ADD | Compute SPW from training data |
| **Cell 19** | MODIFY | Use `OPTIMAL_THRESHOLD` instead of hardcoded `0.5` |
| **Cell 25** | NO CHANGE | Recalibration runs automatically on new model |
| **Cell 32** | MODIFY | Use `OPTIMAL_THRESHOLD` in summary |

---

## Phase 1 — Threshold Sweep (Zero Cost, Run First)

### What to add: New cell after Cell 17

**Purpose:** Find the threshold on validation set that maximises macro F1.
Macro F1 is the right metric here — it treats home wins and away wins equally.
Raw accuracy would keep pushing us toward 0.5 (biased toward majority class).

```python
# ── Phase 1: Threshold Sweep on Validation Set ──────────────────────────────
# Run this BEFORE retraining. Uses existing final_model val_probs.
# We pick threshold on VALIDATION — never on test set.

from sklearn.metrics import (
    f1_score, recall_score, precision_score, fbeta_score
)

# Get validation probabilities from current final_model
val_probs_final = final_model.predict_proba(X_valid)[:, 1]

thresholds  = np.arange(0.38, 0.57, 0.01)
sweep_rows  = []

for t in thresholds:
    preds = (val_probs_final >= t).astype(int)
    away_recall  = recall_score(y_valid, preds, pos_label=0, zero_division=0)
    home_recall  = recall_score(y_valid, preds, pos_label=1, zero_division=0)
    away_prec    = precision_score(y_valid, preds, pos_label=0, zero_division=0)
    home_prec    = precision_score(y_valid, preds, pos_label=1, zero_division=0)
    macro_f1     = f1_score(y_valid, preds, average='macro', zero_division=0)
    accuracy     = accuracy_score(y_valid, preds)
    recall_gap   = abs(home_recall - away_recall)   # lower = more balanced

    sweep_rows.append({
        'threshold'  : round(t, 2),
        'accuracy'   : accuracy,
        'macro_f1'   : macro_f1,
        'away_recall': away_recall,
        'home_recall': home_recall,
        'away_prec'  : away_prec,
        'home_prec'  : home_prec,
        'recall_gap' : recall_gap,
    })

sweep_df = pd.DataFrame(sweep_rows)

# Pick optimal threshold: highest macro_f1 on validation
OPTIMAL_THRESHOLD = sweep_df.loc[sweep_df['macro_f1'].idxmax(), 'threshold']
best_row = sweep_df[sweep_df['threshold'] == OPTIMAL_THRESHOLD].iloc[0]

print('Threshold Sweep — Validation Set')
print('=' * 75)
print(sweep_df[['threshold','accuracy','macro_f1','away_recall',
                'home_recall','recall_gap']].to_string(index=False))
print('=' * 75)
print(f'\nOptimal threshold (max macro F1): {OPTIMAL_THRESHOLD}')
print(f'  Accuracy    : {best_row["accuracy"]:.4f}')
print(f'  Macro F1    : {best_row["macro_f1"]:.4f}')
print(f'  Away Recall : {best_row["away_recall"]:.4f}')
print(f'  Home Recall : {best_row["home_recall"]:.4f}')
print(f'  Recall Gap  : {best_row["recall_gap"]:.4f}')

# Plot the sweep
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
fig.suptitle('Threshold Sweep — Validation Set', fontsize=14, fontweight='bold')

axes[0].plot(sweep_df['threshold'], sweep_df['macro_f1'],
             color='#2196F3', linewidth=2.5, label='Macro F1')
axes[0].axvline(OPTIMAL_THRESHOLD, color='red', linestyle='--',
                label=f'Optimal = {OPTIMAL_THRESHOLD}')
axes[0].set_xlabel('Threshold'); axes[0].set_ylabel('Macro F1')
axes[0].set_title('Macro F1 vs Threshold')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(sweep_df['threshold'], sweep_df['away_recall'],
             color='#FF5722', linewidth=2.5, label='Away Recall')
axes[1].plot(sweep_df['threshold'], sweep_df['home_recall'],
             color='#4CAF50', linewidth=2.5, label='Home Recall')
axes[1].axvline(OPTIMAL_THRESHOLD, color='red', linestyle='--',
                label=f'Optimal = {OPTIMAL_THRESHOLD}')
axes[1].set_xlabel('Threshold'); axes[1].set_ylabel('Recall')
axes[1].set_title('Home vs Away Recall')
axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].plot(sweep_df['threshold'], sweep_df['accuracy'],
             color='#9C27B0', linewidth=2.5, label='Accuracy')
axes[2].axvline(0.50, color='gray', linestyle=':', alpha=0.6, label='Old threshold = 0.5')
axes[2].axvline(OPTIMAL_THRESHOLD, color='red', linestyle='--',
                label=f'New optimal = {OPTIMAL_THRESHOLD}')
axes[2].set_xlabel('Threshold'); axes[2].set_ylabel('Accuracy')
axes[2].set_title('Accuracy vs Threshold')
axes[2].legend(); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../reports/threshold_sweep_v1.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'Chart saved → ../reports/threshold_sweep_v1.png')
```

---

## Phase 2 — scale_pos_weight Retraining

### Step 2A — Compute SPW (New cell before Cell 17, right after Cell 11)

Add this immediately after Cell 11 (where X_train / y_train are defined):

```python
# ── Compute scale_pos_weight ─────────────────────────────────────────────────
# scale_pos_weight = count(negative class) / count(positive class)
# In XGBoost: positive class = 1 = home win
#             negative class = 0 = away win
# This makes the model treat 1 away win as equivalent to SPW home wins during training.

away_wins_count = (y_train == 0).sum()
home_wins_count = (y_train == 1).sum()
SPW = away_wins_count / home_wins_count

print(f'Training set class distribution:')
print(f'  Home wins  : {home_wins_count:,}  ({home_wins_count/len(y_train):.1%})')
print(f'  Away wins  : {away_wins_count:,}  ({away_wins_count/len(y_train):.1%})')
print(f'  Imbalance  : {home_wins_count/away_wins_count:.3f}:1')
print(f'  SPW value  : {SPW:.4f}')
print(f'\nInterpretation: each away win in training now counts as {SPW:.2f} home wins')
```

### Step 2B — Modify Cell 17 (Final Model Training)

**Before (current Cell 17):**
```python
final_model = xgb.XGBClassifier(
    **best_params,
    eval_metric='logloss',
    early_stopping_rounds=40,
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1,
)
```

**After (modified Cell 17):**
```python
# ── Final model with scale_pos_weight to correct home/away bias ──────────────
final_model = xgb.XGBClassifier(
    **best_params,
    scale_pos_weight      = SPW,    # ← NEW: corrects 1.43:1 class imbalance
    eval_metric           = 'logloss',
    early_stopping_rounds = 40,
    use_label_encoder     = False,
    random_state          = 42,
    n_jobs                = -1,
)
final_model.fit(
    X_train, y_train,
    eval_set = [(X_train, y_train), (X_valid, y_valid)],
    verbose  = 100,
)
print(f'Best iteration: {final_model.best_iteration}')
```

**Note on what changes inside XGBoost:**
`scale_pos_weight` multiplies the gradient contribution of positive class (home wins)
by `1/SPW` effectively, making the model penalise away win misses more heavily.
The tree structure changes. The optimal threshold will also shift — it will no longer
be 0.5. This is why we re-run the threshold sweep after retraining.

---

## Phase 3 — Re-run Threshold Sweep on New Model

After Cell 17 has been retrained with SPW, add a second threshold sweep cell:

```python
# ── Phase 3: Threshold Sweep AFTER scale_pos_weight retraining ───────────────
# SPW shifts the model's probability outputs — optimal threshold will change.
# Must re-sweep on validation set to find new optimal.

val_probs_spw = final_model.predict_proba(X_valid)[:, 1]

sweep_rows_spw = []
for t in np.arange(0.38, 0.57, 0.01):
    preds = (val_probs_spw >= t).astype(int)
    sweep_rows_spw.append({
        'threshold'  : round(t, 2),
        'accuracy'   : accuracy_score(y_valid, preds),
        'macro_f1'   : f1_score(y_valid, preds, average='macro', zero_division=0),
        'away_recall': recall_score(y_valid, preds, pos_label=0, zero_division=0),
        'home_recall': recall_score(y_valid, preds, pos_label=1, zero_division=0),
        'recall_gap' : abs(
            recall_score(y_valid, preds, pos_label=1, zero_division=0) -
            recall_score(y_valid, preds, pos_label=0, zero_division=0)
        ),
    })

sweep_spw_df = pd.DataFrame(sweep_rows_spw)
OPTIMAL_THRESHOLD_SPW = sweep_spw_df.loc[
    sweep_spw_df['macro_f1'].idxmax(), 'threshold'
]

# Compare original threshold sweep vs SPW threshold sweep
print('Comparison: Original Model vs SPW Model — Validation Set')
print('=' * 65)
print(f'{"Metric":<25} {"Original":>12} {"With SPW":>12} {"Delta":>10}')
print('-' * 65)

orig_best = sweep_df[sweep_df['threshold'] == OPTIMAL_THRESHOLD].iloc[0]
spw_best  = sweep_spw_df[sweep_spw_df['threshold'] == OPTIMAL_THRESHOLD_SPW].iloc[0]

for metric in ['accuracy', 'macro_f1', 'away_recall', 'home_recall', 'recall_gap']:
    orig_val = orig_best[metric]
    spw_val  = spw_best[metric]
    delta    = spw_val - orig_val
    arrow    = '▲' if delta > 0 else '▼'
    print(f'{metric:<25} {orig_val:>12.4f} {spw_val:>12.4f} {arrow}{abs(delta):>8.4f}')

print('=' * 65)
print(f'\nOriginal optimal threshold : {OPTIMAL_THRESHOLD}')
print(f'SPW model optimal threshold: {OPTIMAL_THRESHOLD_SPW}')

# Lock in the final threshold — use SPW model's optimal
FINAL_THRESHOLD = OPTIMAL_THRESHOLD_SPW
print(f'\nFINAL_THRESHOLD locked at: {FINAL_THRESHOLD} (for all test evaluation below)')
```

---

## Phase 4 — Update Cell 19 (Test Evaluation)

**Before:**
```python
test_preds = (test_probs >= 0.5).astype(int)
```

**After:**
```python
# Use optimal threshold found on validation set — NOT hardcoded 0.5
test_probs = final_model.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= FINAL_THRESHOLD).astype(int)

print(f'Using optimised threshold: {FINAL_THRESHOLD}  (was 0.5)')
```

Everything else in Cell 19 stays the same — accuracy, AUC, log loss, brier,
classification_report. The threshold change will affect accuracy, recall, F1.
AUC and log loss are threshold-independent (they don't change).

---

## Phase 5 — Calibration Automatically Updates (Cell 25)

Cell 25 does not need code changes. It fits on `X_valid, y_valid` using
whatever `final_model` is in scope. After retraining, `final_model` is
the new SPW model. The calibration will refit on the new probability space.

**One change needed:** update the calibration accuracy print to use
`FINAL_THRESHOLD` instead of `0.5`:

**Before:**
```python
print(f'Calibrated accuracy      : {accuracy_score(y_test, (cal_probs>=0.5).astype(int)):.4f}')
```

**After:**
```python
print(f'Calibrated accuracy (t={FINAL_THRESHOLD}) : {accuracy_score(y_test, (cal_probs>=FINAL_THRESHOLD).astype(int)):.4f}')
```

---

## Phase 6 — Update Cell 32 (Summary)

**Before:**
```python
summary_acc = accuracy_score(y_test, (summary_probs >= 0.5).astype(int))
```

**After:**
```python
summary_acc = accuracy_score(y_test, (summary_probs >= FINAL_THRESHOLD).astype(int))

# Also add new lines to the print block:
print(f'  Optimal Threshold: {FINAL_THRESHOLD}  (tuned on validation set)')
print(f'  SPW used         : {SPW:.4f}  (corrects {home_wins_count/away_wins_count:.2f}:1 imbalance)')
```

---

## How to Read the Results — Decision Criteria

After running all phases, you will have a before/after comparison.
Here is how to interpret each metric:

### Accuracy
```
If drops by more than 1pp (e.g. 64.3% → 63.0%) → SPW overcorrected.
  Action: reduce SPW. Try SPW * 0.8 (softer correction).

If roughly flat or better (64.3% → 63.8–65.5%) → successful fix.
  Action: proceed.
```

### ROC-AUC
```
Should stay within ±0.003 of 0.7092.
AUC is threshold-independent — SPW changes training but good signal stays good.

If AUC drops by more than 0.005 → SPW is hurting the model's ranking ability.
  Action: reduce SPW or investigate further.
```

### Away Recall (the primary target metric)
```
Current: 0.4937
Target : 0.60+ (a 10+ pp improvement)

If away recall < 0.55 after all fixes → the fix did not work adequately.
  Action: try scale_pos_weight = 0.5 (more aggressive) or lower threshold further.

If away recall > 0.70 → overcorrected, now missing too many home wins.
  Action: increase threshold or reduce SPW.
```

### Macro F1 (the most important summary metric for balance)
```
Current macro F1 ≈ 0.62  (from classification report: (0.55 + 0.70) / 2)
Target  macro F1 > 0.64

Macro F1 is the right metric for a balanced model.
If it improves → the fix worked. If it drops → something went wrong.
```

### Betting Simulation
```
After fix, expect:
- More away bets (because model now calls away wins more often)
- Slightly lower ROI per bet (because some of the new away bets will be lower-confidence)
- Higher total volume of bets
- Net profit should stay equal or improve

If total profit from simulation drops significantly → the new away bets
are not reliable enough. May need higher minimum edge threshold.
```

---

## Expected Before/After Numbers

Based on typical SPW + threshold tuning results on imbalanced sports data:

| Metric | Before | Expected After | How confident |
|---|---|---|---|
| Accuracy | 64.3% | 63.5–65.5% | Medium |
| ROC-AUC | 0.7092 | 0.706–0.712 | High (stays stable) |
| Away Recall | 0.4937 | 0.58–0.66 | High |
| Home Recall | 0.7562 | 0.68–0.74 | High |
| Macro F1 | ~0.62 | 0.63–0.65 | Medium |
| Optimal threshold | 0.50 | 0.44–0.48 | High (SPW shifts it down) |

---

## Risk Mitigations

**Risk 1: SPW makes model too aggressive on away wins**
- Symptom: away recall jumps to >0.72 but home recall drops below 0.65
- Mitigation: calculated SPW (≈0.698) is a conservative correction.
  If this happens, try `SPW * 0.85` (~0.59) for a gentler push

**Risk 2: Best iteration changes significantly after SPW**
- SPW affects gradient magnitudes, which can change convergence speed
- `early_stopping_rounds=40` handles this automatically
- Monitor: if best_iteration < 300 the model converged too fast (SPW too aggressive)

**Risk 3: Calibration degrades after SPW**
- SPW shifts the raw probability distribution
- Cell 25 refits the calibrator on validation — this is self-correcting
- If Brier score worsens significantly after calibration, check calibration curve plot

**Risk 4: Betting simulation profit drops**
- More away bets at lower confidence can lower win rate
- Mitigation: run simulation at threshold=0.58 and 0.60 first before 0.52
- Only bet away games where model probability is 0.42 or lower (high confidence away)

---

## Exact Execution Order (Step by Step)

```
Step 1  Read Cell 11 output — confirm away_wins and home_wins counts
Step 2  Add SPW computation cell after Cell 11
Step 3  Run Cell 11 + SPW cell — verify SPW ≈ 0.698
Step 4  Add threshold sweep cell after Cell 17 (Phase 1 code above)
Step 5  Run Cells 13 → 17 → threshold sweep — record OPTIMAL_THRESHOLD
Step 6  Modify Cell 17 — add scale_pos_weight = SPW
Step 7  Re-run Cell 17 — note new best_iteration
Step 8  Run Phase 3 threshold sweep cell — record OPTIMAL_THRESHOLD_SPW
Step 9  Update Cell 19 — replace 0.5 with FINAL_THRESHOLD
Step 10 Run Cell 19 — read new test set results
Step 11 Run Cell 20 (confusion matrix + ROC) — no code change needed
Step 12 Run Cell 22 (betting simulation) — note if more away bets appear
Step 13 Run Cell 25 (calibration) — update threshold in print statement
Step 14 Run Cells 26–29 (ROI analysis, feature importance) — no changes needed
Step 15 Update Cell 32 — add FINAL_THRESHOLD and SPW to summary
Step 16 Run Cell 30 (save artifacts) — new models saved with new timestamp
Step 17 Compare before vs after using the decision criteria above
```

---

## What Success Looks Like

The fix is successful if all three of these are true:

1. **Away recall ≥ 0.58** (was 0.49 — minimum 9pp improvement)
2. **Macro F1 ≥ 0.63** (was ~0.62 — directional improvement)
3. **ROC-AUC ≥ 0.705** (was 0.709 — max acceptable drop of 0.004)

If all three pass → update the team report, mark Action 1 complete, move to Action 2.

If any fail → use the risk mitigations above, do not move to Action 2 until Action 1 is stable.

---

*Plan based on direct code reading of training.ipynb cells 8, 10, 11, 13, 15, 17, 19, 22, 25, 30, 32*  
*All cell numbers refer to the current training.ipynb cell index (0-based)*

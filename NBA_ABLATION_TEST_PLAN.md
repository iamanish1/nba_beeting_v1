# Ablation Test — Implementation Plan
**Feature Under Investigation:** `implied_prob_home` and ELO feature redundancy  
**Notebook:** `notebooks/ablation_test.ipynb` (new)  
**Estimated time:** 3–4 hours end to end

---

## What We Discovered Before Writing This Plan

From reading the source code directly:

**`implied_prob_home` is NOT from betting odds.**

It is calculated in `src/pipeline/elo.py` lines 133–136 using this formula:

```python
def _expected_win(elo_home, elo_away, home_adv=100.0):
    return 1.0 / (1.0 + 10 ** (-(elo_home + home_adv - elo_away) / 400.0))

df["implied_prob_home"] = df.apply(
    lambda r: _expected_win(r["elo_home"], r["elo_away"], HOME_ADV_ELO),
    axis=1,
)
```

**There is zero data leakage from betting markets.**
`line_movement` (the actual betting column) is `np.nan` for all 29,575 games —
it is an unfilled placeholder in `pipeline.py` line 182.

---

## What We Are Actually Testing Now

Since there is no leakage, the ablation test shifts to a different but equally
important question:

> **Are we wasting model capacity on redundant ELO features?**

Your feature set currently contains these ELO-related columns:

| Feature | Source | What it is |
|---|---|---|
| `elo_home` | elo.py line 128 | Raw ELO rating of home team |
| `elo_away` | elo.py line 129 | Raw ELO rating of away team |
| `elo_difference` | elo.py line 130 | `elo_home - elo_away` (linear) |
| `implied_prob_home` | elo.py line 133 | Non-linear transform of elo_home & elo_away |
| `elo_rolling_five_home` | elo.py | 5-game rolling ELO — home |
| `elo_rolling_five_away` | elo.py | 5-game rolling ELO — away |
| `elo_diff` | training.ipynb cell 10 | `elo_home - elo_away` (differential feature) |

**The redundancy problem:**

`elo_difference` and `elo_diff` are **identical** — both equal `elo_home - elo_away`.
One of them must be removed.

`implied_prob_home` is a **non-linear transformation** of `elo_home` and `elo_away`
(with +100 home advantage added before the transform).
Since `elo_home` and `elo_away` are already in the feature set, the model can learn
the non-linear relationship itself. But giving it explicitly may help convergence.

The ablation test will tell us which ELO representation is most efficient.

---

## Test Design — 5 Experiments

We will train 5 versions of the model, each with a different ELO feature set.
**Everything else stays identical:** same hyperparameters, same train/val/test split,
same random seed, same early stopping, same evaluation code.

| Experiment | ELO Features Kept | Purpose |
|---|---|---|
| **E0 — Current (baseline)** | All 7 ELO features including duplicates | Reproduce current 63.82% as control |
| **E1 — Remove duplicate** | Remove `elo_difference` (keep `elo_diff`) | Fix the known duplicate, measure effect |
| **E2 — Remove implied_prob** | E1 minus `implied_prob_home` | Does non-linear ELO transform add value? |
| **E3 — Raw ELO only** | Keep only `elo_home`, `elo_away`, rolling × 2 | Minimal ELO representation |
| **E4 — Probability only** | Keep only `implied_prob_home`, rolling × 2 | Does probability beat raw ratings? |

**Decision rule after running all 5:**
- Pick the experiment with highest **ROC-AUC on test set**
- Use that feature configuration going forward
- If E0 wins → keep all (no change, but now we know why)
- If E2 wins → drop both duplicates and implied_prob
- If E1 wins → just drop the duplicate, keep implied_prob

---

## Implementation — Step by Step

### Step 0 — Create a New Notebook

Do not modify `training.ipynb`. Create a clean new file:
`notebooks/ablation_test.ipynb`

This protects the original results. The ablation notebook runs independently.

---

### Step 1 — Setup Cell (Copy from training.ipynb)

```python
import sys, os, warnings, joblib
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss,
    brier_score_loss, classification_report
)

RANDOM_STATE = 42
print("Libraries loaded.")
```

---

### Step 2 — Load Data (Exact Same as training.ipynb)

```python
MASTER_PATH = '../data/master_dataset.csv'
master = pd.read_csv(MASTER_PATH, parse_dates=['game_date'])
master = master.sort_values('game_date').reset_index(drop=True)

print(f"Loaded: {master.shape[0]:,} rows x {master.shape[1]} columns")
print(f"Date range: {master['game_date'].min()} to {master['game_date'].max()}")
```

---

### Step 3 — Exact Same Split as training.ipynb

**Critical: use the identical split boundaries. Never shuffle.**

```python
TRAIN_END   = 2021
VALID_START = 2022
VALID_END   = 2023
TEST_START  = 2024

train_df = master[master['season'] <= TRAIN_END].copy()
valid_df = master[(master['season'] >= VALID_START) & (master['season'] <= VALID_END)].copy()
test_df  = master[master['season'] >= TEST_START].copy()

print(f"Train : {len(train_df):,} rows")
print(f"Valid : {len(valid_df):,} rows")
print(f"Test  : {len(test_df):,} rows")

# Verify these match training.ipynb exactly
assert len(train_df) == 25929, f"Train size mismatch: {len(train_df)}"
assert len(valid_df) == 2767,  f"Valid size mismatch: {len(valid_df)}"
assert len(test_df)  == 879,   f"Test size mismatch: {len(test_df)}"
print("Split sizes verified — match training.ipynb exactly.")
```

---

### Step 4 — Reproduce Feature Engineering (Exact Same as training.ipynb)

```python
# Exact same DROP_COLS and FEATURE_COLS as training.ipynb
DROP_COLS = ['game_id', 'game_date', 'season', 'home_team_id', 'away_team_id',
             'home_win', 'line_movement']
FEATURE_COLS = [c for c in master.columns if c not in DROP_COLS]

# Exact same differential feature function
def add_differential_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    diff_pairs = [
        ('elo_home',                         'elo_away'),
        ('net_rating_home',                  'net_rating_away'),
        ('offensive_rating_home',            'offensive_rating_away'),
        ('defensive_rating_home',            'defensive_rating_away'),
        ('last_5_win_rate_home',             'last_5_win_rate_away'),
        ('last_10_win_rate_home',            'last_10_win_rate_away'),
        ('ppg_10_home',                      'ppg_10_away'),
        ('rest_days_home',                   'rest_days_away'),
        ('fatigue_load_index_home',          'fatigue_load_index_away'),
        ('turnovers_per_game_home',          'turnovers_per_game_away'),
        ('player_impact_estimate_home',      'player_impact_estimate_away'),
        ('injured_count_home',               'injured_count_away'),
        ('coaching_adaptability_score_home', 'coaching_adaptability_score_away'),
    ]
    for col_a, col_b in diff_pairs:
        if col_a in df.columns and col_b in df.columns:
            name = col_a.replace('_home', '') + '_diff'
            df[name] = df[col_a] - df[col_b]
    return df

master   = add_differential_features(master)
train_df = add_differential_features(train_df)
valid_df = add_differential_features(valid_df)
test_df  = add_differential_features(test_df)

# Build ALL_FEATURES with deduplication (the fix from training.ipynb)
DIFF_FEATURES = [c for c in master.columns if c.endswith('_diff')]
ALL_FEATURES  = list(dict.fromkeys(FEATURE_COLS + DIFF_FEATURES))

# Remove low variance
var_threshold = 1e-4
low_var = [c for c in ALL_FEATURES if c in train_df.columns
           and train_df[c].var() < var_threshold]
ALL_FEATURES = [c for c in ALL_FEATURES if c not in low_var]

print(f"Total features: {len(ALL_FEATURES)}")

# Verify feature count matches training.ipynb
assert len(ALL_FEATURES) == 69, f"Feature count mismatch: {len(ALL_FEATURES)}"
print("Feature count verified — matches training.ipynb exactly.")
```

---

### Step 5 — Define the 5 Experiment Feature Sets

```python
# Identify all ELO-related features present
elo_features_present = [f for f in ALL_FEATURES if 'elo' in f.lower()]
print("ELO features in ALL_FEATURES:", elo_features_present)
# Expected: ['elo_home', 'elo_away', 'elo_difference', 'elo_rolling_five_home',
#            'elo_rolling_five_away', 'implied_prob_home', 'elo_diff']

# ── Define experiment feature sets ──────────────────────────────────────────

# E0: Current state — all features including known duplicate
FEATURES_E0 = ALL_FEATURES.copy()

# E1: Remove elo_difference (exact duplicate of elo_diff)
FEATURES_E1 = [f for f in ALL_FEATURES if f != 'elo_difference']

# E2: Remove both elo_difference AND implied_prob_home
FEATURES_E2 = [f for f in ALL_FEATURES if f not in ('elo_difference', 'implied_prob_home')]

# E3: Remove elo_difference, implied_prob_home, AND raw elo_home/elo_away
#     Keep only elo_diff and rolling ELO
FEATURES_E3 = [f for f in ALL_FEATURES
               if f not in ('elo_difference', 'implied_prob_home', 'elo_home', 'elo_away')]

# E4: Remove elo_difference, elo_diff, raw elo_home/elo_away
#     Keep only implied_prob_home and rolling ELO (probability representation only)
FEATURES_E4 = [f for f in ALL_FEATURES
               if f not in ('elo_difference', 'elo_diff', 'elo_home', 'elo_away')]

experiments = {
    'E0_current':         FEATURES_E0,
    'E1_remove_dup':      FEATURES_E1,
    'E2_remove_implied':  FEATURES_E2,
    'E3_raw_elo_only':    FEATURES_E3,
    'E4_prob_only':       FEATURES_E4,
}

print("\nExperiment feature counts:")
for name, feats in experiments.items():
    elo_in = [f for f in feats if 'elo' in f.lower() or 'implied' in f.lower()]
    print(f"  {name}: {len(feats)} features | ELO cols: {elo_in}")
```

---

### Step 6 — Build X/y Splits for Each Experiment

```python
def build_splits(feature_list):
    """Build train/valid/test X matrices for a given feature list."""
    # Verify all features exist in the dataframes
    missing = [f for f in feature_list if f not in train_df.columns]
    if missing:
        raise ValueError(f"Features missing from train_df: {missing}")

    X_tr = train_df[feature_list].fillna(0)
    y_tr = train_df['home_win']

    X_va = valid_df[feature_list].fillna(0)
    y_va = valid_df['home_win']

    X_te = test_df[feature_list].fillna(0)
    y_te = test_df['home_win']

    # Sanity check: no duplicate columns
    assert X_tr.columns.duplicated().sum() == 0, "Duplicate columns detected!"
    assert X_tr.isnull().sum().sum() == 0,       "NaN values remain after fillna!"

    return X_tr, y_tr, X_va, y_va, X_te, y_te

print("build_splits() function defined.")
```

---

### Step 7 — Define the Model (Exact Same Params as Final Model in training.ipynb)

```python
# These are the best_params found by RandomizedSearchCV in training.ipynb
# DO NOT change these — we are testing features, not model config
BEST_PARAMS = {
    'subsample'        : 0.8,
    'reg_lambda'       : 0.5,
    'reg_alpha'        : 0.01,
    'n_estimators'     : 2000,      # high ceiling, early stopping will find optimum
    'min_child_weight' : 3,
    'max_depth'        : 4,
    'learning_rate'    : 0.01,
    'gamma'            : 0.5,
    'colsample_bytree' : 0.8,
}

def train_experiment(X_tr, y_tr, X_va, y_va, experiment_name):
    """Train one XGBoost model. Returns fitted model."""
    print(f"\n{'='*55}")
    print(f"  Training: {experiment_name} | Features: {X_tr.shape[1]}")
    print(f"{'='*55}")

    model = xgb.XGBClassifier(
        **BEST_PARAMS,
        eval_metric          = 'logloss',
        early_stopping_rounds= 40,
        use_label_encoder    = False,
        random_state         = RANDOM_STATE,
        n_jobs               = -1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set = [(X_tr, y_tr), (X_va, y_va)],
        verbose  = 200,
    )
    print(f"  Best iteration: {model.best_iteration}")
    return model

print("train_experiment() function defined.")
```

---

### Step 8 — Define Evaluation (Identical Metrics to training.ipynb)

```python
def evaluate_experiment(model, X_te, y_te, experiment_name):
    """Evaluate on test set. Returns dict of metrics."""
    probs = model.predict_proba(X_te)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        'experiment'   : experiment_name,
        'n_features'   : X_te.shape[1],
        'best_iter'    : model.best_iteration,
        'accuracy'     : accuracy_score(y_te, preds),
        'roc_auc'      : roc_auc_score(y_te, probs),
        'log_loss'     : log_loss(y_te, probs),
        'brier'        : brier_score_loss(y_te, probs),
        'away_recall'  : 0.0,  # filled below
        'home_recall'  : 0.0,  # filled below
    }

    # Per-class recall (the home/away imbalance metric)
    report = classification_report(y_te, preds,
                                   target_names=['Away Win', 'Home Win'],
                                   output_dict=True)
    metrics['away_recall'] = report['Away Win']['recall']
    metrics['home_recall'] = report['Home Win']['recall']

    print(f"\n  Results — {experiment_name}")
    print(f"  Accuracy   : {metrics['accuracy']:.4f}")
    print(f"  ROC-AUC    : {metrics['roc_auc']:.4f}")
    print(f"  Log Loss   : {metrics['log_loss']:.4f}")
    print(f"  Brier      : {metrics['brier']:.4f}")
    print(f"  Away Recall: {metrics['away_recall']:.4f}")
    print(f"  Home Recall: {metrics['home_recall']:.4f}")

    return metrics

print("evaluate_experiment() function defined.")
```

---

### Step 9 — Run All 5 Experiments

```python
# Storage for results
all_results = []
trained_models = {}

# Run each experiment
for exp_name, feature_list in experiments.items():

    # Build splits for this feature set
    X_tr, y_tr, X_va, y_va, X_te, y_te = build_splits(feature_list)

    # Train
    model = train_experiment(X_tr, y_tr, X_va, y_va, exp_name)

    # Evaluate
    metrics = evaluate_experiment(model, X_te, y_te, exp_name)

    # Store
    all_results.append(metrics)
    trained_models[exp_name] = model

print("\n\nAll experiments complete.")
```

---

### Step 10 — Build Results Comparison Table

```python
results_df = pd.DataFrame(all_results).set_index('experiment')

# Add delta columns vs E0 baseline
results_df['acc_delta']  = results_df['accuracy'] - results_df.loc['E0_current', 'accuracy']
results_df['auc_delta']  = results_df['roc_auc']  - results_df.loc['E0_current', 'roc_auc']

# Format for display
display_cols = ['n_features', 'accuracy', 'acc_delta', 'roc_auc',
                'auc_delta', 'log_loss', 'brier', 'away_recall', 'home_recall', 'best_iter']

print("\n" + "="*90)
print("  ABLATION TEST RESULTS — COMPLETE COMPARISON")
print("="*90)
print(results_df[display_cols].to_string(
    float_format=lambda x: f"{x:+.4f}" if abs(x) < 0.1 else f"{x:.4f}"
))
print("="*90)

# Highlight best experiment by AUC (primary metric)
best_exp = results_df['roc_auc'].idxmax()
print(f"\nBest experiment by ROC-AUC : {best_exp}")
print(f"Best experiment by Accuracy: {results_df['accuracy'].idxmax()}")
```

---

### Step 11 — Visualise the Results

```python
fig, axes = plt.subplots(2, 3, figsize=(20, 11))
fig.suptitle('Ablation Test Results — ELO Feature Variants', fontsize=15, fontweight='bold')

exp_names  = results_df.index.tolist()
short_names = ['E0\nCurrent', 'E1\nRemove Dup', 'E2\nNo Implied',
               'E3\nRaw ELO', 'E4\nProb Only']
colors = ['#90CAF9', '#A5D6A7', '#FFCC80', '#EF9A9A', '#CE93D8']

def bar_chart(ax, values, title, ylabel, baseline_val=None):
    bars = ax.bar(short_names, values, color=colors, edgecolor='white', linewidth=1.2)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel)
    if baseline_val is not None:
        ax.axhline(baseline_val, color='red', linestyle='--', linewidth=1.2,
                   label=f'E0 baseline = {baseline_val:.4f}')
        ax.legend(fontsize=8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylim(min(values)*0.995, max(values)*1.01)

e0_acc = results_df.loc['E0_current', 'accuracy']
e0_auc = results_df.loc['E0_current', 'roc_auc']

bar_chart(axes[0,0], results_df['accuracy'].values,    'Accuracy (Higher = Better)',  'Accuracy',  e0_acc)
bar_chart(axes[0,1], results_df['roc_auc'].values,     'ROC-AUC (Higher = Better)',   'ROC-AUC',   e0_auc)
bar_chart(axes[0,2], results_df['log_loss'].values,    'Log Loss (Lower = Better)',   'Log Loss')
bar_chart(axes[1,0], results_df['brier'].values,       'Brier Score (Lower = Better)','Brier')
bar_chart(axes[1,1], results_df['away_recall'].values, 'Away Win Recall',             'Recall')
bar_chart(axes[1,2], results_df['home_recall'].values, 'Home Win Recall',             'Recall')

plt.tight_layout()
plt.savefig('../reports/ablation_elo_features.png', dpi=150, bbox_inches='tight')
plt.show()
print("Chart saved.")
```

---

### Step 12 — Feature Count vs Performance Plot

```python
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(results_df['n_features'], results_df['roc_auc'],
           s=200, color=colors, edgecolors='black', linewidths=1.5, zorder=5)

for idx, (name, row) in enumerate(results_df.iterrows()):
    ax.annotate(name,
                (row['n_features'], row['roc_auc']),
                textcoords='offset points', xytext=(8, 4), fontsize=9)

ax.set_xlabel('Number of Features', fontsize=12)
ax.set_ylabel('ROC-AUC on Test Set', fontsize=12)
ax.set_title('Feature Count vs Model Performance\n(goal: same AUC with fewer features)',
             fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

### Step 13 — Final Decision Cell

```python
print("="*60)
print("  ABLATION TEST DECISION")
print("="*60)

best_by_auc = results_df['roc_auc'].idxmax()
auc_improvement = results_df.loc[best_by_auc, 'auc_delta']

print(f"\nBest configuration  : {best_by_auc}")
print(f"AUC vs E0 baseline  : {auc_improvement:+.4f}")
print(f"Features removed    : {results_df.loc['E0_current', 'n_features'] - results_df.loc[best_by_auc, 'n_features']}")

print("\n--- Interpretation Guide ---")
print("""
  auc_delta > +0.002 → The reduced feature set GENUINELY IMPROVES the model.
                        Switch to that configuration immediately.

  auc_delta 0.000 to +0.002 → Effectively equal performance.
                               Prefer the simpler model (fewer features = less noise).
                               Switch to that configuration.

  auc_delta -0.001 to 0.000 → Tiny drop. Negligible.
                               If the simpler config, still prefer it.

  auc_delta < -0.001 → The removed features were contributing real signal.
                        Keep the fuller feature set (or investigate which
                        specific feature was most responsible).
""")

# Auto-recommend
if auc_improvement >= -0.001:
    print(f"RECOMMENDATION: Use {best_by_auc} going forward.")
    best_features = experiments[best_by_auc]
    print(f"  Feature count: {len(best_features)}")
    elo_kept = [f for f in best_features if 'elo' in f.lower() or 'implied' in f.lower()]
    print(f"  ELO features kept: {elo_kept}")
else:
    print("RECOMMENDATION: Keep E0_current — removed features are contributing.")
```

---

## Checklist Before Running

Go through this before executing the notebook:

- [ ] New notebook created at `notebooks/ablation_test.ipynb`
- [ ] `master_dataset.csv` accessible at `../data/master_dataset.csv`
- [ ] All imports load without error (Step 1)
- [ ] Data loads correctly — 29,575 rows (Step 2)
- [ ] Split size assertions pass — 25929 / 2767 / 879 (Step 3)
- [ ] Feature count assertion passes — 69 features (Step 4)
- [ ] ELO feature list printed in Step 5 — verify 7 features listed
- [ ] `build_splits()` sanity checks pass (no duplicates, no NaN) (Step 6)
- [ ] `BEST_PARAMS` matches exactly what `training.ipynb` used (Step 7)
- [ ] Reports directory exists at `../reports/` or create it before Step 11

---

## What to Do With the Results

### If E1 wins (just remove the duplicate `elo_difference`):
Update `training.ipynb` cell 10 — add `elo_difference` to the filter after `ALL_FEATURES` is built:
```python
ALL_FEATURES = [f for f in ALL_FEATURES if f != 'elo_difference']
```

### If E2 wins (remove duplicate + implied_prob):
Add both to the filter:
```python
ALL_FEATURES = [f for f in ALL_FEATURES
                if f not in ('elo_difference', 'implied_prob_home')]
```

### If E3 or E4 wins (more aggressive ELO reduction):
Add the corresponding list to the filter in cell 10.

### If E0 wins (current is best):
No change to training.ipynb needed.
Document the result: the non-linear ELO probability transform plus the duplicate
both contribute independently. Move on to the next improvement.

---

## After the Ablation Test

Once you have the winning feature configuration, the next steps in order are:

1. Apply the winning config to `training.ipynb` — retrain final model
2. Re-run betting simulation to get updated ROI numbers
3. Begin Priority 2: star player out features (the highest-ceiling improvement)

---

*Plan based on direct source code reading of elo.py, pipeline.py, features.py*  
*No assumptions — all findings verified from actual code*

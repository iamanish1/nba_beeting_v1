"""
train_ensemble.py
-----------------
Reproducible temporal training script for the NBA betting model.

What this script does:
1. Loads master_dataset.csv
2. Rebuilds differential features used in the notebooks
3. Adds missing indicators for sparse market + semantic-missing features
4. Trains an XGBoost baseline plus sklearn ensemble members
5. Learns a logistic stacker on validation predictions
6. Reports validation/test metrics and approximate ROI by confidence threshold

Usage:
    .\\venv\\Scripts\\python.exe src\\modeling\\train_ensemble.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Allow running from project root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.modeling.common import (
    SEMANTIC_MISSING_COLS,
    SPARSE_MARKET_COLS,
    add_differential_features,
    add_missing_indicators,
    approximate_home_roi,
    build_feature_columns,
    build_mlp_pipeline,
    build_temporal_splits,
    evaluate_predictions,
    load_master_dataset,
)

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "xgboost is required for this script. Run it with the project venv:\n"
        ".\\venv\\Scripts\\python.exe src\\modeling\\train_ensemble.py"
    ) from exc


def _report_metrics(name: str, metrics: dict[str, float]) -> None:
    print(
        f"{name:<18}"
        f" acc={metrics['accuracy']:.4f}"
        f" auc={metrics['roc_auc']:.4f}"
        f" logloss={metrics['log_loss']:.4f}"
        f" brier={metrics['brier']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train stacked NBA ensemble.")
    parser.add_argument("--master-path", default="data/master_dataset.csv")
    parser.add_argument("--artifacts-dir", default="models/ensemble")
    parser.add_argument("--train-end", type=int, default=2021)
    parser.add_argument("--valid-start", type=int, default=2022)
    parser.add_argument("--valid-end", type=int, default=2023)
    parser.add_argument("--test-start", type=int, default=2024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    master = load_master_dataset(args.master_path)
    master = add_differential_features(master)
    master = add_missing_indicators(master, SPARSE_MARKET_COLS)
    master = add_missing_indicators(master, SEMANTIC_MISSING_COLS)

    splits = build_temporal_splits(
        master,
        train_end=args.train_end,
        valid_start=args.valid_start,
        valid_end=args.valid_end,
        test_start=args.test_start,
    )
    feature_cols = build_feature_columns(splits.train)

    print(f"Rows   train={len(splits.train):,} valid={len(splits.valid):,} test={len(splits.test):,}")
    print(f"Features selected: {len(feature_cols)}")
    print("Pipeline priorities audit:")
    print(f"  H2H present: {'h2h_home_win_rate_10' in master.columns}")
    print(f"  Home-court rolling strength present: {'home_court_strength_home' in master.columns}")
    print(f"  Sparse market missing flags added: {sum(col.endswith('_missing') for col in feature_cols)}")

    X_train = splits.train[feature_cols].copy()
    y_train = splits.train["home_win"].astype(int)
    X_valid = splits.valid[feature_cols].copy()
    y_valid = splits.valid["home_win"].astype(int)
    X_test = splits.test[feature_cols].copy()
    y_test = splits.test["home_win"].astype(int)

    xgb = XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=40,
        random_state=args.seed,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    hgb = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=4,
        max_iter=300,
        min_samples_leaf=20,
        l2_regularization=0.1,
        early_stopping=True,
        random_state=args.seed,
    )
    hgb.fit(X_train, y_train)

    mlp = build_mlp_pipeline(random_state=args.seed)
    mlp.fit(X_train, y_train)

    valid_pred_table = pd.DataFrame(
        {
            "xgb": xgb.predict_proba(X_valid)[:, 1],
            "hgb": hgb.predict_proba(X_valid)[:, 1],
            "mlp": mlp.predict_proba(X_valid)[:, 1],
        },
        index=X_valid.index,
    )
    test_pred_table = pd.DataFrame(
        {
            "xgb": xgb.predict_proba(X_test)[:, 1],
            "hgb": hgb.predict_proba(X_test)[:, 1],
            "mlp": mlp.predict_proba(X_test)[:, 1],
        },
        index=X_test.index,
    )

    stacker = LogisticRegression(max_iter=1000, random_state=args.seed)
    stacker.fit(valid_pred_table, y_valid)

    valid_metrics = {
        "xgb": evaluate_predictions(y_valid, valid_pred_table["xgb"].to_numpy()),
        "hgb": evaluate_predictions(y_valid, valid_pred_table["hgb"].to_numpy()),
        "mlp": evaluate_predictions(y_valid, valid_pred_table["mlp"].to_numpy()),
    }
    valid_metrics["stack"] = evaluate_predictions(
        y_valid,
        stacker.predict_proba(valid_pred_table)[:, 1],
    )

    test_metrics = {
        "xgb": evaluate_predictions(y_test, test_pred_table["xgb"].to_numpy()),
        "hgb": evaluate_predictions(y_test, test_pred_table["hgb"].to_numpy()),
        "mlp": evaluate_predictions(y_test, test_pred_table["mlp"].to_numpy()),
    }
    stack_test_probs = stacker.predict_proba(test_pred_table)[:, 1]
    test_metrics["stack"] = evaluate_predictions(y_test, stack_test_probs)

    print("\nValidation metrics")
    for name, metrics in valid_metrics.items():
        _report_metrics(name, metrics)

    print("\nTest metrics")
    for name, metrics in test_metrics.items():
        _report_metrics(name, metrics)

    roi_rows = approximate_home_roi(splits.test, stack_test_probs)
    if roi_rows:
        print("\nApproximate home-side ROI (using closing implied probability as market proxy)")
        for row in roi_rows:
            if row["bets"] == 0:
                print(f"  threshold={row['threshold']:.2f} bets=0")
                continue
            print(
                f"  threshold={row['threshold']:.2f}"
                f" bets={row['bets']}"
                f" win_rate={row['win_rate']:.4f}"
                f" roi={row['roi']:.4f}"
            )

    summary = {
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "roi": roi_rows,
        "splits": {
            "train_end": args.train_end,
            "valid_start": args.valid_start,
            "valid_end": args.valid_end,
            "test_start": args.test_start,
        },
    }

    (artifacts_dir / "ensemble_summary.json").write_text(json.dumps(summary, indent=2))
    joblib.dump(hgb, artifacts_dir / "histgb.joblib")
    joblib.dump(mlp, artifacts_dir / "mlp.joblib")
    joblib.dump(stacker, artifacts_dir / "stacker.joblib")
    joblib.dump(feature_cols, artifacts_dir / "feature_columns.joblib")
    xgb.save_model(artifacts_dir / "xgb.json")
    test_pred_table.assign(stack=stack_test_probs, y_true=y_test.values).to_csv(
        artifacts_dir / "test_predictions.csv",
        index=False,
    )

    print(f"\nSaved artifacts to {artifacts_dir}")


if __name__ == "__main__":
    main()

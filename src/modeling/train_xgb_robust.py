from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

# Allow running from project root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.modeling.common import (
    DENSE_MARKET_COLS,
    EXTERNAL_INJURY_COLS,
    EXTERNAL_LINEUP_COLS,
    H2H_STRUCTURAL_COLS,
    H2H_MISSING_FLAG_COLS,
    LOCAL_LINEUP_COLS,
    NEW_SIGNAL_FAMILY_COLS,
    SEMANTIC_MISSING_COLS,
    SPARSE_MARKET_COLS,
    add_differential_features,
    add_missing_indicators,
    apply_neutral_feature_defaults,
    approximate_home_roi,
    build_feature_columns_for_mode,
    build_sample_weights,
    build_temporal_folds,
    build_temporal_splits,
    compute_reciprocal_class_weight,
    evaluate_predictions,
    load_master_dataset,
    optimize_threshold,
)

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "xgboost is required for this script. Run it with the project venv:\n"
        ".\\venv\\Scripts\\python.exe src\\modeling\\train_xgb_robust.py"
    ) from exc


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    return {
        col: float(frame[col].mean())
        for col in frame.columns
        if pd.api.types.is_numeric_dtype(frame[col])
    }


def _build_xgb(seed: int, away_bias: bool, scale_pos_weight: float) -> XGBClassifier:
    return _build_xgb_config(
        seed=seed,
        away_bias=away_bias,
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=40,
    )


def _build_xgb_config(
    *,
    seed: int,
    away_bias: bool,
    scale_pos_weight: float,
    early_stopping_rounds: int,
) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.5,
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=early_stopping_rounds,
        random_state=seed,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight if away_bias else 1.0,
    )


def _build_notebook_feature_columns(master: pd.DataFrame, train_df: pd.DataFrame) -> list[str]:
    drop_cols = ["game_id", "game_date", "season", "home_team_id", "away_team_id", "home_win"]
    feature_cols = [c for c in master.columns if c not in drop_cols]
    diff_features = [c for c in master.columns if c.endswith("_diff")]
    all_features = list(dict.fromkeys(feature_cols + diff_features))

    # The notebook only appends sparse-market missing flags explicitly.
    sparse_flags = [f"{col}_missing" for col in SPARSE_MARKET_COLS if f"{col}_missing" in train_df.columns]
    all_features = list(dict.fromkeys(all_features + sparse_flags))

    # Keep the notebook's sparse-market flags, but remove the extra dense-market
    # flags added for the robust harness and the new H2H missing flags.
    dense_only_flags = [
        f"{col}_missing"
        for col in DENSE_MARKET_COLS
        if col != "home_spread_close" and f"{col}_missing" in train_df.columns
    ]
    banned = set(dense_only_flags + H2H_MISSING_FLAG_COLS)
    all_features = [c for c in all_features if c not in banned]

    low_var = [col for col in all_features if col in train_df.columns and train_df[col].var(skipna=True) < 1e-4]
    all_features = [c for c in all_features if c not in set(NEW_SIGNAL_FAMILY_COLS)]
    return [col for col in all_features if col not in low_var]


def _fit_and_score_candidate(
    *,
    name: str,
    master: pd.DataFrame,
    feature_mode: str,
    include_h2h: bool,
    include_sparse_market: bool,
    recency_decay: float | None,
    away_bias: bool,
    seed: int,
    threshold_objective: str = "macro_recall",
    feature_strategy: str = "mode",
    early_stopping_rounds: int = 40,
    eval_with_train: bool = False,
    extra_features: list[str] | None = None,
    drop_features: list[str] | None = None,
    feature_families: list[str] | None = None,
) -> dict[str, object]:
    splits = build_temporal_splits(master)
    if feature_strategy == "notebook":
        feature_cols = _build_notebook_feature_columns(master, splits.train)
    else:
        feature_cols = build_feature_columns_for_mode(splits.train, mode=feature_mode)

    if not include_h2h:
        feature_cols = [c for c in feature_cols if c not in H2H_STRUCTURAL_COLS and not c.startswith("h2h_")]

    if not include_sparse_market:
        banned = set(SPARSE_MARKET_COLS)
        feature_cols = [c for c in feature_cols if c not in banned]

    if extra_features:
        feature_cols = list(dict.fromkeys(feature_cols + [c for c in extra_features if c in splits.train.columns]))

    if drop_features:
        banned = set(drop_features)
        feature_cols = [c for c in feature_cols if c not in banned]

    family_map = {
        "local_lineup": LOCAL_LINEUP_COLS,
        "external_injury": EXTERNAL_INJURY_COLS,
        "external_lineup": EXTERNAL_LINEUP_COLS,
        "market_missing_flags": [f"{col}_missing" for col in DENSE_MARKET_COLS + SPARSE_MARKET_COLS],
    }
    for family in feature_families or []:
        feature_cols = list(dict.fromkeys(feature_cols + [c for c in family_map.get(family, []) if c in splits.train.columns]))

    X_train = splits.train[feature_cols].copy()
    y_train = splits.train["home_win"].astype(int)
    X_valid = splits.valid[feature_cols].copy()
    y_valid = splits.valid["home_win"].astype(int)
    X_test = splits.test[feature_cols].copy()
    y_test = splits.test["home_win"].astype(int)

    class_weight = compute_reciprocal_class_weight(y_train)
    sample_weight = build_sample_weights(splits.train, decay=recency_decay)

    model = _build_xgb_config(
        seed=seed,
        away_bias=away_bias,
        scale_pos_weight=class_weight,
        early_stopping_rounds=early_stopping_rounds,
    )
    eval_set = [(X_valid, y_valid)]
    if eval_with_train:
        eval_set = [(X_train, y_train), (X_valid, y_valid)]
    fit_kwargs = {"eval_set": eval_set, "verbose": False}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight

    model.fit(X_train, y_train, **fit_kwargs)

    calibrator = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    calibrator.fit(X_valid, y_valid)

    valid_probs = calibrator.predict_proba(X_valid)[:, 1]
    threshold, threshold_sweep = optimize_threshold(y_valid, valid_probs, objective=threshold_objective)

    test_probs = calibrator.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_predictions(y_test, test_probs, threshold=threshold)
    valid_metrics = evaluate_predictions(y_valid, valid_probs, threshold=threshold)
    roi_rows = approximate_home_roi(splits.test, test_probs)

    fold_rows: list[dict[str, float]] = []
    for fold in build_temporal_folds(master, first_test_season=2022):
        fold_features = [c for c in feature_cols if c in fold.train.columns]
        fold_model = _build_xgb_config(
            seed=seed,
            away_bias=away_bias,
            scale_pos_weight=compute_reciprocal_class_weight(fold.train["home_win"]),
            early_stopping_rounds=early_stopping_rounds,
        )
        fold_eval_set = [(fold.valid[fold_features], fold.valid["home_win"].astype(int))]
        if eval_with_train:
            fold_eval_set = [
                (fold.train[fold_features], fold.train["home_win"].astype(int)),
                (fold.valid[fold_features], fold.valid["home_win"].astype(int)),
            ]
        fold_fit_kwargs = {"eval_set": fold_eval_set, "verbose": False}
        fold_weights = build_sample_weights(fold.train, decay=recency_decay)
        if fold_weights is not None:
            fold_fit_kwargs["sample_weight"] = fold_weights

        fold_model.fit(
            fold.train[fold_features],
            fold.train["home_win"].astype(int),
            **fold_fit_kwargs,
        )
        fold_cal = CalibratedClassifierCV(FrozenEstimator(fold_model), method="isotonic")
        fold_cal.fit(fold.valid[fold_features], fold.valid["home_win"].astype(int))
        fold_valid_probs = fold_cal.predict_proba(fold.valid[fold_features])[:, 1]
        fold_threshold, _ = optimize_threshold(
            fold.valid["home_win"],
            fold_valid_probs,
            objective=threshold_objective,
        )
        fold_test_probs = fold_cal.predict_proba(fold.test[fold_features])[:, 1]
        fold_metrics = evaluate_predictions(fold.test["home_win"], fold_test_probs, threshold=fold_threshold)
        fold_metrics["season"] = int(fold.test["season"].iloc[0])
        fold_rows.append(fold_metrics)

    return {
        "name": name,
        "feature_count": len(feature_cols),
        "feature_mode": feature_mode,
        "include_h2h": include_h2h,
        "include_sparse_market": include_sparse_market,
        "recency_decay": recency_decay,
        "away_bias": away_bias,
        "threshold_objective": threshold_objective,
        "feature_strategy": feature_strategy,
        "threshold": threshold,
        "feature_columns": feature_cols,
        "feature_families": feature_families or [],
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "rolling_metrics_mean": _mean_metrics(fold_rows),
        "rolling_metrics_by_season": fold_rows,
        "roi": roi_rows,
        "xgb_best_iteration": int(getattr(model, "best_iteration", -1)),
        "threshold_sweep_top": threshold_sweep.sort_values(
            by=(
                ["macro_f1", "accuracy", "away_recall"]
                if threshold_objective == "macro_f1"
                else ["macro_recall", "accuracy", "away_recall"]
            ),
            ascending=[False, False, False],
        ).head(5).to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Robust XGBoost experiments for NBA betting model.")
    parser.add_argument("--master-path", default="data/master_dataset.csv")
    parser.add_argument("--artifacts-dir", default="models/xgb_robust")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    master = load_master_dataset(args.master_path)
    master = apply_neutral_feature_defaults(master)
    master = add_differential_features(master)
    master = add_missing_indicators(master, DENSE_MARKET_COLS)
    master = add_missing_indicators(master, SPARSE_MARKET_COLS)
    master = add_missing_indicators(master, SEMANTIC_MISSING_COLS)

    candidates = [
        {
            "name": "notebook_xgb_baseline",
            "feature_mode": "all",
            "include_h2h": True,
            "include_sparse_market": True,
            "recency_decay": None,
            "away_bias": True,
            "threshold_objective": "macro_f1",
            "feature_strategy": "notebook",
            "early_stopping_rounds": 60,
            "eval_with_train": True,
            "extra_features": None,
            "drop_features": None,
            "feature_families": [],
        },
        {
            "name": "baseline_plus_market_coverage_fix",
            "feature_mode": "all",
            "include_h2h": True,
            "include_sparse_market": True,
            "recency_decay": None,
            "away_bias": True,
            "threshold_objective": "macro_f1",
            "feature_strategy": "notebook",
            "early_stopping_rounds": 60,
            "eval_with_train": True,
            "extra_features": None,
            "drop_features": None,
            "feature_families": ["market_missing_flags"],
        },
        {
            "name": "baseline_plus_local_lineup_features",
            "feature_mode": "all",
            "include_h2h": True,
            "include_sparse_market": True,
            "recency_decay": None,
            "away_bias": True,
            "threshold_objective": "macro_f1",
            "feature_strategy": "notebook",
            "early_stopping_rounds": 60,
            "eval_with_train": True,
            "extra_features": None,
            "drop_features": None,
            "feature_families": ["local_lineup"],
        },
        {
            "name": "baseline_plus_external_injury_features",
            "feature_mode": "all",
            "include_h2h": True,
            "include_sparse_market": True,
            "recency_decay": None,
            "away_bias": True,
            "threshold_objective": "macro_f1",
            "feature_strategy": "notebook",
            "early_stopping_rounds": 60,
            "eval_with_train": True,
            "extra_features": None,
            "drop_features": None,
            "feature_families": ["external_injury"],
        },
        {
            "name": "baseline_plus_external_lineup_features",
            "feature_mode": "all",
            "include_h2h": True,
            "include_sparse_market": True,
            "recency_decay": None,
            "away_bias": True,
            "threshold_objective": "macro_f1",
            "feature_strategy": "notebook",
            "early_stopping_rounds": 60,
            "eval_with_train": True,
            "extra_features": None,
            "drop_features": None,
            "feature_families": ["external_lineup"],
        },
        {
            "name": "baseline_plus_market_and_injuries",
            "feature_mode": "all",
            "include_h2h": True,
            "include_sparse_market": True,
            "recency_decay": None,
            "away_bias": True,
            "threshold_objective": "macro_f1",
            "feature_strategy": "notebook",
            "early_stopping_rounds": 60,
            "eval_with_train": True,
            "extra_features": None,
            "drop_features": None,
            "feature_families": ["market_missing_flags", "local_lineup", "external_injury"],
        },
        {
            "name": "baseline_plus_market_injuries_and_lineups",
            "feature_mode": "all",
            "include_h2h": True,
            "include_sparse_market": True,
            "recency_decay": None,
            "away_bias": True,
            "threshold_objective": "macro_f1",
            "feature_strategy": "notebook",
            "early_stopping_rounds": 60,
            "eval_with_train": True,
            "extra_features": None,
            "drop_features": None,
            "feature_families": ["market_missing_flags", "local_lineup", "external_injury", "external_lineup"],
        },
        {
            "name": "baseline_plus_opening_lines",
            "feature_mode": "all",
            "include_h2h": True,
            "include_sparse_market": True,
            "recency_decay": None,
            "away_bias": True,
            "threshold_objective": "macro_f1",
            "feature_strategy": "notebook",
            "early_stopping_rounds": 60,
            "eval_with_train": True,
            "extra_features": [
                "open_implied_prob_home",
                "spread_movement_pts",
                "line_movement",
                "open_implied_prob_home_missing",
                "spread_movement_pts_missing",
                "line_movement_missing",
            ],
            "drop_features": None,
            "feature_families": [],
        },
    ]

    results: list[dict[str, object]] = []
    for candidate in candidates:
        print(f"\n=== Running {candidate['name']} ===")
        result = _fit_and_score_candidate(master=master, seed=args.seed, **candidate)
        results.append(result)
        test_metrics = result["test_metrics"]
        rolling = result["rolling_metrics_mean"]
        print(
            "test"
            f" acc={test_metrics['accuracy']:.4f}"
            f" auc={test_metrics['roc_auc']:.4f}"
            f" brier={test_metrics['brier']:.4f}"
            f" away_recall={test_metrics['away_recall']:.4f}"
            f" home_recall={test_metrics['home_recall']:.4f}"
            f" threshold={result['threshold']:.2f}"
        )
        if rolling:
            print(
                "rolling"
                f" acc={rolling.get('accuracy', float('nan')):.4f}"
                f" auc={rolling.get('roc_auc', float('nan')):.4f}"
                f" away_recall={rolling.get('away_recall', float('nan')):.4f}"
                f" home_recall={rolling.get('home_recall', float('nan')):.4f}"
            )

    summary_df = pd.DataFrame(
        [
            {
                "name": row["name"],
                "feature_count": row["feature_count"],
                "threshold": row["threshold"],
                "test_accuracy": row["test_metrics"]["accuracy"],
                "test_auc": row["test_metrics"]["roc_auc"],
                "test_brier": row["test_metrics"]["brier"],
                "test_away_recall": row["test_metrics"]["away_recall"],
                "test_home_recall": row["test_metrics"]["home_recall"],
                "rolling_accuracy": row["rolling_metrics_mean"].get("accuracy", np.nan),
                "rolling_auc": row["rolling_metrics_mean"].get("roc_auc", np.nan),
                "rolling_away_recall": row["rolling_metrics_mean"].get("away_recall", np.nan),
                "rolling_home_recall": row["rolling_metrics_mean"].get("home_recall", np.nan),
            }
            for row in results
        ]
    ).sort_values(["test_accuracy", "rolling_accuracy"], ascending=False)

    summary_df.to_csv(artifacts_dir / "experiment_summary.csv", index=False)
    (artifacts_dir / "experiment_summary.json").write_text(json.dumps(results, indent=2))

    feature_inventory_dir = artifacts_dir / "feature_inventory"
    feature_inventory_dir.mkdir(parents=True, exist_ok=True)
    for row in results:
        (feature_inventory_dir / f"{row['name']}.json").write_text(
            json.dumps(
                {
                    "name": row["name"],
                    "feature_count": row["feature_count"],
                    "feature_families": row.get("feature_families", []),
                    "feature_columns": row.get("feature_columns", []),
                },
                indent=2,
            )
        )

    baseline = next((row for row in results if row["name"] == "notebook_xgb_baseline"), None)
    promotion_rows = []
    for row in results:
        promote = False
        reason = "baseline_control"
        if baseline and row["name"] != "notebook_xgb_baseline":
            test_gain = row["test_metrics"]["accuracy"] - baseline["test_metrics"]["accuracy"]
            rolling_delta = row["rolling_metrics_mean"].get("accuracy", np.nan) - baseline["rolling_metrics_mean"].get("accuracy", np.nan)
            recall_gap = abs(row["test_metrics"]["home_recall"] - row["test_metrics"]["away_recall"])
            baseline_gap = abs(baseline["test_metrics"]["home_recall"] - baseline["test_metrics"]["away_recall"])
            promote = test_gain >= 0.002 and (np.isnan(rolling_delta) or rolling_delta >= -0.005) and recall_gap <= baseline_gap + 0.05
            reason = "promote" if promote else "hold"
            promotion_rows.append(
                {
                    "name": row["name"],
                    "test_gain_vs_baseline": test_gain,
                    "rolling_gain_vs_baseline": rolling_delta,
                    "recall_gap": recall_gap,
                    "baseline_recall_gap": baseline_gap,
                    "promotion_decision": reason,
                }
            )
    if promotion_rows:
        pd.DataFrame(promotion_rows).to_csv(artifacts_dir / "promotion_report.csv", index=False)

    best_name = summary_df.iloc[0]["name"] if not summary_df.empty else "n/a"
    print(f"\nSaved experiment summary to {artifacts_dir}")
    print(f"Best candidate by test/rolling sort: {best_name}")


if __name__ == "__main__":
    main()

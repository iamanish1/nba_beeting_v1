from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


DROP_COLS = [
    "game_id",
    "game_date",
    "season",
    "home_team_id",
    "away_team_id",
    "home_win",
]

DIFF_PAIRS = [
    ("elo_home", "elo_away"),
    ("net_rating_home", "net_rating_away"),
    ("offensive_rating_home", "offensive_rating_away"),
    ("defensive_rating_home", "defensive_rating_away"),
    ("last_5_win_rate_home", "last_5_win_rate_away"),
    ("last_10_win_rate_home", "last_10_win_rate_away"),
    ("ppg_10_home", "ppg_10_away"),
    ("rest_days_home", "rest_days_away"),
    ("fatigue_load_index_home", "fatigue_load_index_away"),
    ("turnovers_per_game_home", "turnovers_per_game_away"),
    ("player_impact_estimate_home", "player_impact_estimate_away"),
    ("injured_count_home", "injured_count_away"),
    ("coaching_adaptability_score_home", "coaching_adaptability_score_away"),
    ("star_points_lost_home", "star_points_lost_away"),
    ("season_pressure_home", "season_pressure_away"),
    ("win_streak_home", "win_streak_away"),
    ("ewm_win_rate_5_home", "ewm_win_rate_5_away"),
    ("injury_impact_score_home", "injury_impact_score_away"),
    ("starter_impact_lost_home", "starter_impact_lost_away"),
    ("lineup_continuity_5_home", "lineup_continuity_5_away"),
    ("expected_starter_stability_home", "expected_starter_stability_away"),
    ("pregame_injury_impact_home", "pregame_injury_impact_away"),
    ("confirmed_starters_available_home", "confirmed_starters_available_away"),
]

SPARSE_MARKET_COLS = [
    "sharp_signal_home",
    "book_consensus_std",
    "spread_movement_pts",
    "open_implied_prob_home",
    "line_movement",
]

DENSE_MARKET_COLS = [
    "home_implied_prob_close",
    "market_elo_diff",
    "home_spread_close",
]

H2H_STRUCTURAL_COLS = [
    "h2h_home_win_rate_10",
    "h2h_home_pts_diff_10",
]

H2H_MISSING_FLAG_COLS = [
    "h2h_home_win_rate_10_missing",
    "h2h_home_pts_diff_10_missing",
]

MARKET_MISSING_FLAG_COLS = [
    "sharp_signal_home_missing",
    "book_consensus_std_missing",
    "spread_movement_pts_missing",
    "open_implied_prob_home_missing",
    "line_movement_missing",
    "home_spread_close_missing",
]

SPARSE_MARKET_FLAG_COLS = [
    "sharp_signal_home_missing",
    "book_consensus_std_missing",
    "spread_movement_pts_missing",
    "open_implied_prob_home_missing",
    "line_movement_missing",
]

SEMANTIC_MISSING_COLS = [
    "coaching_adaptability_score_home",
    "coaching_adaptability_score_away",
    "star_points_lost_home",
    "star_points_lost_away",
]

LOCAL_LINEUP_COLS = [
    "starters_available_home",
    "starters_available_away",
    "starters_missing_home",
    "starters_missing_away",
    "top5_minutes_missing_home",
    "top5_minutes_missing_away",
    "top3_scorers_missing_home",
    "top3_scorers_missing_away",
    "starter_minutes_share_lost_home",
    "starter_minutes_share_lost_away",
    "rotation_minutes_share_lost_home",
    "rotation_minutes_share_lost_away",
    "scoring_share_lost_home",
    "scoring_share_lost_away",
    "injury_impact_score_home",
    "injury_impact_score_away",
    "starter_impact_lost_home",
    "starter_impact_lost_away",
    "lineup_continuity_3_home",
    "lineup_continuity_3_away",
    "lineup_continuity_5_home",
    "lineup_continuity_5_away",
    "lineup_continuity_10_home",
    "lineup_continuity_10_away",
    "returning_starter_count_home",
    "returning_starter_count_away",
    "expected_starter_stability_home",
    "expected_starter_stability_away",
]

EXTERNAL_INJURY_COLS = [
    "questionable_count_home",
    "questionable_count_away",
    "doubtful_count_home",
    "doubtful_count_away",
    "out_count_home",
    "out_count_away",
    "pregame_injury_impact_home",
    "pregame_injury_impact_away",
    "external_injury_reports_present_home",
    "external_injury_reports_present_away",
]

EXTERNAL_LINEUP_COLS = [
    "confirmed_starters_available_home",
    "confirmed_starters_available_away",
    "confirmed_starters_missing_home",
    "confirmed_starters_missing_away",
    "lineup_confirmation_lag_hours_home",
    "lineup_confirmation_lag_hours_away",
    "external_lineups_present_home",
    "external_lineups_present_away",
]

NEW_SIGNAL_FAMILY_COLS = LOCAL_LINEUP_COLS + EXTERNAL_INJURY_COLS + EXTERNAL_LINEUP_COLS


@dataclass
class SplitBundle:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


@dataclass
class TemporalFold:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    label: str


def load_master_dataset(path: str | Path) -> pd.DataFrame:
    master = pd.read_csv(path, parse_dates=["game_date"])
    return master.sort_values("game_date").reset_index(drop=True)


def add_differential_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for home_col, away_col in DIFF_PAIRS:
        if home_col in df.columns and away_col in df.columns:
            diff_name = home_col.replace("_home", "") + "_diff"
            df[diff_name] = df[home_col] - df[away_col]
    return df


def add_missing_indicators(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            flag = f"{col}_missing"
            if flag not in df.columns:
                df[flag] = df[col].isna().astype(int)
    return df


def apply_neutral_feature_defaults(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    neutral_defaults = {
        "h2h_home_win_rate_10": 0.5,
        "h2h_home_pts_diff_10": 0.0,
    }
    for col, default in neutral_defaults.items():
        if col in df.columns:
            flag = f"{col}_missing"
            if flag not in df.columns:
                df[flag] = df[col].isna().astype(int)
            df[col] = df[col].fillna(default)
    return df


def build_temporal_splits(
    master: pd.DataFrame,
    train_end: int = 2021,
    valid_start: int = 2022,
    valid_end: int = 2023,
    test_start: int = 2024,
) -> SplitBundle:
    return SplitBundle(
        train=master[master["season"] <= train_end].copy(),
        valid=master[(master["season"] >= valid_start) & (master["season"] <= valid_end)].copy(),
        test=master[master["season"] >= test_start].copy(),
    )


def build_feature_columns(df: pd.DataFrame, low_var_threshold: float = 1e-4) -> list[str]:
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    usable = df[feature_cols]
    low_var = [c for c in usable.columns if usable[c].var(skipna=True) < low_var_threshold]
    return [c for c in feature_cols if c not in low_var]

def build_feature_columns_for_mode(df: pd.DataFrame, mode: str = "all") -> list[str]:
    feature_cols = build_feature_columns(df)
    if mode == "all":
        return feature_cols
    if mode == "dense_market":
        banned = set(SPARSE_MARKET_COLS + SPARSE_MARKET_FLAG_COLS)
        return [c for c in feature_cols if c not in banned]
    if mode == "no_market_sparse":
        banned = set(SPARSE_MARKET_COLS + SPARSE_MARKET_FLAG_COLS)
        return [c for c in feature_cols if c not in banned]
    if mode == "no_h2h":
        banned = set(H2H_STRUCTURAL_COLS + H2H_MISSING_FLAG_COLS)
        return [c for c in feature_cols if c not in banned]
    raise ValueError(f"Unknown feature mode: {mode}")


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=int)
    preds = (probs >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true_arr, preds)),
        "roc_auc": float(roc_auc_score(y_true_arr, probs)),
        "log_loss": float(log_loss(y_true_arr, probs, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true_arr, probs)),
        "home_recall": float(recall_score(y_true_arr, preds, pos_label=1, zero_division=0)),
        "away_recall": float(recall_score(y_true_arr, preds, pos_label=0, zero_division=0)),
        "home_precision": float(precision_score(y_true_arr, preds, pos_label=1, zero_division=0)),
        "away_precision": float(precision_score(y_true_arr, preds, pos_label=0, zero_division=0)),
        "macro_recall": float(recall_score(y_true_arr, preds, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true_arr, preds, average="macro", zero_division=0)),
        "threshold": float(threshold),
    }


def optimize_threshold(
    y_true: pd.Series | np.ndarray,
    probs: np.ndarray,
    thresholds: Iterable[float] = np.arange(0.40, 0.61, 0.01),
    objective: str = "macro_recall",
) -> tuple[float, pd.DataFrame]:
    y_true_arr = np.asarray(y_true, dtype=int)
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        metrics = evaluate_predictions(y_true_arr, probs, threshold=float(threshold))
        metrics["recall_gap"] = abs(metrics["home_recall"] - metrics["away_recall"])
        rows.append(metrics)
    sweep = pd.DataFrame(rows)
    sort_specs = {
        "macro_recall": (["macro_recall", "accuracy", "away_recall"], [False, False, False]),
        "macro_f1": (["macro_f1", "accuracy", "away_recall"], [False, False, False]),
    }
    if objective not in sort_specs:
        raise ValueError(f"Unknown threshold objective: {objective}")
    sort_by, ascending = sort_specs[objective]
    best = sweep.sort_values(by=sort_by, ascending=ascending).iloc[0]
    return float(best["threshold"]), sweep


def compute_reciprocal_class_weight(y: pd.Series | np.ndarray) -> float:
    y_arr = np.asarray(y, dtype=int)
    home_wins = max(int((y_arr == 1).sum()), 1)
    away_wins = max(int((y_arr == 0).sum()), 1)
    return away_wins / home_wins


def build_sample_weights(df: pd.DataFrame, decay: float | None = None) -> np.ndarray | None:
    if decay is None or decay <= 0:
        return None
    seasons = df["season"].to_numpy(dtype=float)
    max_season = seasons.max()
    return np.exp(-decay * (max_season - seasons))


def build_temporal_folds(
    master: pd.DataFrame,
    train_span: int = 15,
    valid_span: int = 2,
    first_test_season: int = 2021,
) -> list[TemporalFold]:
    seasons = sorted(int(s) for s in master["season"].dropna().unique())
    folds: list[TemporalFold] = []
    for test_season in seasons:
        if test_season < first_test_season:
            continue
        valid_end = test_season - 1
        valid_start = valid_end - valid_span + 1
        train_end = valid_start - 1
        train_start = max(seasons[0], train_end - train_span + 1)
        train = master[(master["season"] >= train_start) & (master["season"] <= train_end)].copy()
        valid = master[(master["season"] >= valid_start) & (master["season"] <= valid_end)].copy()
        test = master[master["season"] == test_season].copy()
        if train.empty or valid.empty or test.empty:
            continue
        folds.append(
            TemporalFold(
                train=train,
                valid=valid,
                test=test,
                label=f"train_{train_start}_{train_end}__valid_{valid_start}_{valid_end}__test_{test_season}",
            )
        )
    return folds


def build_mlp_pipeline(random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    alpha=1e-3,
                    learning_rate_init=1e-3,
                    max_iter=300,
                    early_stopping=True,
                    n_iter_no_change=20,
                    random_state=random_state,
                ),
            ),
        ]
    )


def approximate_home_roi(
    df: pd.DataFrame,
    probs: np.ndarray,
    thresholds: Iterable[float] = (0.52, 0.55, 0.58, 0.60),
) -> list[dict[str, float]]:
    if "home_implied_prob_close" not in df.columns:
        return []

    market_prob = df["home_implied_prob_close"].to_numpy(dtype=float)
    y_true = df["home_win"].to_numpy(dtype=int)
    rows: list[dict[str, float]] = []

    valid_market = np.isfinite(market_prob) & (market_prob > 0) & (market_prob < 1)
    decimal_odds = np.where(valid_market, 1.0 / market_prob, np.nan)

    for threshold in thresholds:
        bet_mask = (probs >= threshold) & valid_market
        if not np.any(bet_mask):
            rows.append(
                {
                    "threshold": threshold,
                    "bets": 0,
                    "accuracy": np.nan,
                    "win_rate": np.nan,
                    "roi": np.nan,
                }
            )
            continue

        profits = np.where(y_true[bet_mask] == 1, decimal_odds[bet_mask] - 1.0, -1.0)
        bet_preds = (probs[bet_mask] >= 0.5).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "bets": int(bet_mask.sum()),
                "accuracy": float(np.mean(bet_preds == y_true[bet_mask])),
                "win_rate": float(np.mean(y_true[bet_mask] == 1)),
                "roi": float(np.mean(profits)),
            }
        )
    return rows

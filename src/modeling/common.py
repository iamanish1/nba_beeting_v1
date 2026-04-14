from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
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
]

SPARSE_MARKET_COLS = [
    "sharp_signal_home",
    "book_consensus_std",
    "spread_movement_pts",
    "open_implied_prob_home",
    "line_movement",
    "home_spread_close",
]

SEMANTIC_MISSING_COLS = [
    "coaching_adaptability_score_home",
    "coaching_adaptability_score_away",
    "star_points_lost_home",
    "star_points_lost_away",
]


@dataclass
class SplitBundle:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


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


def evaluate_predictions(y_true: pd.Series, probs: np.ndarray) -> dict[str, float]:
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "log_loss": float(log_loss(y_true, probs, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, probs)),
    }


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

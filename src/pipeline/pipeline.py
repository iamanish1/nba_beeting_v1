"""
pipeline.py
-----------
Master dataset builder.

Orchestrates: ingest → ELO → features → merge → output

Usage:
    from src.pipeline.pipeline import build_master_dataset
    master = build_master_dataset()
    master.to_csv("data/master_dataset.csv", index=False)

Output columns (one row per game, home-team perspective):
    game_id, game_date, season,
    home_team_id, away_team_id,
    is_home,                       ← always 1 (home team row)
    home_win  ← TARGET

    # ELO
    elo_home, elo_away, elo_difference, elo_rolling_five_home, elo_rolling_five_away

    # Rolling performance (home)
    off_rating_home, def_rating_home, net_rating_home,
    ppg_home, tpg_home, fg_pct_home,
    last_5_win_rate_home, last_10_win_rate_home

    # Rolling performance (away)
    off_rating_away, def_rating_away, net_rating_away,
    ppg_away, tpg_away, fg_pct_away,
    last_5_win_rate_away, last_10_win_rate_away

    # Opponent context (from opponent's past games)
    opp_off_rating_10_home, opp_def_rating_10_home
    opp_off_rating_10_away, opp_def_rating_10_away

    # Game style
    pace_difference, field_goal_difference, shooting_pct_home, shooting_pct_away

    # Player impact
    pie_home, pie_away
    injury_flag_home, injury_flag_away
    injured_count_home, injured_count_away
    star_available_home, star_available_away

    # Betting market proxy
    implied_prob_home      ← ELO-based (real line_movement = NaN placeholder)
    line_movement          ← NaN (fill from betting API)

    # Fatigue & scheduling
    rest_days_home, rest_days_away
    back_to_back_home, back_to_back_away
    fatigue_load_home, fatigue_load_away

    # Meta
    coaching_score_home, coaching_score_away
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from .ingest   import load_all
from .elo      import compute_elo, compute_elo_rolling_five
from .features import (
    build_team_game_log,
    compute_possession_stats,
    compute_pie,
    compute_star_availability,
    add_rolling_performance,
    add_possession_rolling,
    add_rest_fatigue,
    add_coaching_score,
    add_opponent_context,
    add_player_features,
    add_star_features,
    add_game_style_features,
)

_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


# ============================================================================
# Public API
# ============================================================================

def build_master_dataset(
    data_dir: Optional[str] = None,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build the complete master dataset for XGBoost training.

    Parameters
    ----------
    data_dir    : path to folder containing the 5 raw CSVs
                  (defaults to  <project_root>/data/)
    output_path : if provided, saves master CSV to this path
    verbose     : print progress steps

    Returns
    -------
    master : pd.DataFrame, one row per game, all features engineered
    """
    base = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR

    # ------------------------------------------------------------------ #
    # 1. INGEST                                                            #
    # ------------------------------------------------------------------ #
    _log("Loading raw data …", verbose)
    raw = load_all(base)
    games   = raw["games"]
    details = raw["details"]
    _log(f"  games={len(games):,}  details={len(details):,}", verbose)

    # ------------------------------------------------------------------ #
    # 2. ELO                                                               #
    # ------------------------------------------------------------------ #
    _log("Computing ELO ratings …", verbose)
    games = compute_elo(games)
    games = compute_elo_rolling_five(games)

    # ------------------------------------------------------------------ #
    # 3. POSSESSION STATS  (from game details)                            #
    # ------------------------------------------------------------------ #
    _log("Aggregating possession stats …", verbose)
    poss_raw   = compute_possession_stats(details)
    poss_stats = compute_pie(poss_raw)

    # ------------------------------------------------------------------ #
    # 4. STAR AVAILABILITY                                                 #
    # ------------------------------------------------------------------ #
    _log("Computing star player availability …", verbose)
    star_avail = compute_star_availability(details)

    # ------------------------------------------------------------------ #
    # 5. TEAM GAME LOG + ALL ROLLING FEATURES                             #
    # ------------------------------------------------------------------ #
    _log("Building team game log …", verbose)
    log = build_team_game_log(games)

    _log("Adding rolling performance features …", verbose)
    log = add_rolling_performance(log)

    _log("Adding possession-based ratings …", verbose)
    log = add_possession_rolling(log, poss_stats)

    _log("Adding rest & fatigue features …", verbose)
    log = add_rest_fatigue(log)

    _log("Adding coaching adaptability score …", verbose)
    log = add_coaching_score(log)

    _log("Adding opponent context features …", verbose)
    log = add_opponent_context(log)

    _log("Adding player injury features …", verbose)
    log = add_player_features(log, poss_stats)

    _log("Adding star availability features …", verbose)
    log = add_star_features(log, star_avail)

    # ------------------------------------------------------------------ #
    # 6. GAME-STYLE FEATURES (game-level, on games_df)                   #
    # ------------------------------------------------------------------ #
    _log("Adding game style features …", verbose)
    games = add_game_style_features(games, poss_stats)

    # ------------------------------------------------------------------ #
    # 7. PIVOT BACK TO WIDE FORMAT (one row per game)                     #
    # ------------------------------------------------------------------ #
    _log("Assembling master dataset …", verbose)
    master = _assemble_wide(games, log)

    # ------------------------------------------------------------------ #
    # 8. BETTING MARKET PLACEHOLDER                                        #
    # ------------------------------------------------------------------ #
    # implied_prob_home already in games (from ELO)
    # line_movement requires external betting API — placeholder NaN
    master["line_movement"] = np.nan

    # ------------------------------------------------------------------ #
    # 9. CLEAN UP & VALIDATE                                               #
    # ------------------------------------------------------------------ #
    master = _validate_and_clean(master)

    _log(f"\nMaster dataset: {len(master):,} rows × {len(master.columns)} columns", verbose)

    if output_path:
        master.to_csv(output_path, index=False)
        _log(f"Saved to: {output_path}", verbose)

    return master


# ============================================================================
# Internal helpers
# ============================================================================

def _assemble_wide(games_df: pd.DataFrame, log: pd.DataFrame) -> pd.DataFrame:
    """
    Join home-team features and away-team features onto the games_df.

    home features  → suffix _home
    away features  → suffix _away
    """
    TEAM_FEATURES = [
        "game_id", "team_id",
        # rolling performance
        "ppg_10", "opp_ppg_10",
        "fg_pct_roll_10", "ft_pct_roll_10", "fg3_pct_roll_10",
        "last_5_win_rate", "last_10_win_rate",
        "offensive_rating", "defensive_rating", "net_rating",
        "turnovers_per_game",
        # opponent context
        "opponent_off_rating_10", "opponent_def_rating_10",
        # player impact
        "player_impact_estimate",
        "player_injury_flag", "injured_count",
        "star_available", "star_count",
        # fatigue
        "rest_days", "back_to_back", "fatigue_load_index",
        # coaching
        "coaching_adaptability_score",
    ]

    # Keep only columns that exist in log
    avail = [c for c in TEAM_FEATURES if c in log.columns]

    home_log = log[log["is_home"] == 1][avail].copy()
    away_log = log[log["is_home"] == 0][avail].copy()

    # Rename feature columns with _home / _away suffix
    feature_cols = [c for c in avail if c not in ("game_id", "team_id")]
    home_log = home_log.rename(columns={c: f"{c}_home" for c in feature_cols})
    away_log = away_log.rename(columns={c: f"{c}_away" for c in feature_cols})

    # Build base game frame
    base = games_df[[
        "GAME_ID", "GAME_DATE_EST", "SEASON",
        "HOME_TEAM_ID", "VISITOR_TEAM_ID",
        "HOME_TEAM_WINS",
        # ELO
        "elo_home", "elo_away", "elo_difference",
        "elo_rolling_five_home", "elo_rolling_five_away",
        "implied_prob_home",
        # Style (already on games_df)
        "pace_home", "pace_away", "pace_difference",
        "field_goal_difference",
        "shooting_pct_home", "shooting_pct_away",
    ]].copy()

    base = base.rename(columns={
        "GAME_ID":          "game_id",
        "GAME_DATE_EST":    "game_date",
        "SEASON":           "season",
        "HOME_TEAM_ID":     "home_team_id",
        "VISITOR_TEAM_ID":  "away_team_id",
        "HOME_TEAM_WINS":   "home_win",
    })

    master = base.merge(
        home_log.rename(columns={"team_id": "home_team_id"}),
        on=["game_id", "home_team_id"], how="left",
    )
    master = master.merge(
        away_log.rename(columns={"team_id": "away_team_id"}),
        on=["game_id", "away_team_id"], how="left",
    )

    return master


def _validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final cleaning:
    - Drop rows where target is null
    - Drop games with insufficient history (first 10 games of any team's season)
      identified by last_10_win_rate being NaN for the home team
    - Cap extreme values
    - Reorder columns logically
    """
    # Drop rows without target
    df = df.dropna(subset=["home_win"])

    # Drop rows where core rolling features haven't warmed up yet
    df = df.dropna(subset=["last_10_win_rate_home", "last_10_win_rate_away"])

    # Cap ELO difference (rare outliers after long win/loss streaks)
    df["elo_difference"] = df["elo_difference"].clip(-400, 400)

    # Ensure bool columns are int
    for col in df.select_dtypes("bool").columns:
        df[col] = df[col].astype(int)

    # Sort by date
    df = df.sort_values("game_date").reset_index(drop=True)

    return df


def _log(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg)


# ============================================================================
# Incremental update
# ============================================================================

def update_master_dataset(
    existing_master_path: str,
    data_dir: Optional[str] = None,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Incrementally update an existing master dataset with new games.

    Strategy:
        1. Rebuild from scratch (ELO must be recomputed sequentially anyway).
        2. For production at scale, cache ELO state after last game and
           only recompute new rows.

    This is the recommended approach for daily updates until the dataset
    grows beyond ~100k games (not an issue for NBA data).
    """
    _log("Rebuilding master dataset with new data …", verbose)
    base = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
    out  = output_path or existing_master_path
    return build_master_dataset(data_dir=base, output_path=out, verbose=verbose)

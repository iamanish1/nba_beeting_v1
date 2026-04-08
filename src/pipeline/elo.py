"""
elo.py
------
Implements a proper NBA ELO rating system.

Design:
  - Every team starts each franchise history at 1500.
  - K-factor = 20  (standard for NBA regular season).
  - Home-court advantage = +100 ELO points added to home team's
    expected win probability (not to their rating directly).
  - End-of-season mean regression: new_rating = 0.75 * final_rating + 0.25 * 1500
    This reflects roster turnover and prevents ratings from diverging too far.

Data leakage prevention:
  - elo_before_game is stored BEFORE the game result is applied.
  - The feature used in the model is always the pre-game ELO, never post-game.
"""

import pandas as pd
import numpy as np

ELO_START    = 1500.0
K_FACTOR     = 20.0
HOME_ADV_ELO = 100.0   # added to home team's rating for E(win) calculation
SEASON_REGRESS = 0.75  # how much of prior-season ELO carries over


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def _expected_win(rating_a: float, rating_b: float, home_adv: float = 0.0) -> float:
    """
    Expected win probability for team A against team B.
    home_adv: extra ELO points for team A if they are playing at home.
    """
    return 1.0 / (1.0 + 10 ** (-(rating_a + home_adv - rating_b) / 400.0))


def _update_elo(rating_a: float, rating_b: float,
                score_a: int, home_adv: float = 0.0) -> tuple:
    """
    Update ELO ratings after a single game.

    Parameters
    ----------
    rating_a : current ELO of team A (home)
    rating_b : current ELO of team B (away)
    score_a  : 1 if team A won, 0 if team B won
    home_adv : ELO advantage for team A (0 if neutral site)

    Returns
    -------
    (new_rating_a, new_rating_b)
    """
    e_a = _expected_win(rating_a, rating_b, home_adv)
    e_b = 1.0 - e_a

    new_a = rating_a + K_FACTOR * (score_a - e_a)
    new_b = rating_b + K_FACTOR * ((1 - score_a) - e_b)
    return new_a, new_b


# ---------------------------------------------------------------------------
# Main ELO builder
# ---------------------------------------------------------------------------

def compute_elo(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pre-game ELO ratings for every game in games_df.

    Processes games in chronological order. At each season boundary,
    all team ratings are regressed toward the mean.

    Parameters
    ----------
    games_df : DataFrame from ingest.load_games()
               Must be sorted ascending by GAME_DATE_EST.

    Returns
    -------
    games_df with these new columns added (pre-game, no leakage):
        elo_home          – home team ELO before the game
        elo_away          – away team ELO before the game
        elo_difference    – elo_home - elo_away  (positive = home favoured)
        implied_prob_home – win probability implied by ELO
    """
    df = games_df.copy().sort_values("GAME_DATE_EST").reset_index(drop=True)

    ratings: dict[int, float] = {}   # team_id -> current ELO
    current_season = None

    elo_home_list   = []
    elo_away_list   = []

    for _, row in df.iterrows():
        home_id = int(row["HOME_TEAM_ID"])
        away_id = int(row["VISITOR_TEAM_ID"])
        season  = row["SEASON"]

        # --- Season boundary: regress ratings toward mean ---
        if season != current_season:
            if current_season is not None:
                for tid in list(ratings.keys()):
                    ratings[tid] = SEASON_REGRESS * ratings[tid] + (1 - SEASON_REGRESS) * ELO_START
            current_season = season

        # Initialise new teams
        if home_id not in ratings:
            ratings[home_id] = ELO_START
        if away_id not in ratings:
            ratings[away_id] = ELO_START

        # --- Record PRE-GAME ELO (no leakage) ---
        elo_home_list.append(ratings[home_id])
        elo_away_list.append(ratings[away_id])

        # --- Update ratings with actual result ---
        home_win = int(row["HOME_TEAM_WINS"])
        new_home, new_away = _update_elo(
            ratings[home_id], ratings[away_id],
            score_a=home_win,
            home_adv=HOME_ADV_ELO,
        )
        ratings[home_id] = new_home
        ratings[away_id] = new_away

    df["elo_home"]       = elo_home_list
    df["elo_away"]       = elo_away_list
    df["elo_difference"] = df["elo_home"] - df["elo_away"]

    # ELO-implied win probability for home team
    df["implied_prob_home"] = df.apply(
        lambda r: _expected_win(r["elo_home"], r["elo_away"], HOME_ADV_ELO),
        axis=1,
    )

    return df


def compute_elo_rolling_five(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add elo_rolling_five_home / elo_rolling_five_away:
    the average of a team's ELO over their last 5 games (pre-game, no leakage).

    Requires elo_home / elo_away columns from compute_elo().
    """
    df = games_df.copy().sort_values("GAME_DATE_EST").reset_index(drop=True)

    # Build long format: team_id → [game_date, pre_game_elo]
    home_elo = df[["GAME_DATE_EST", "HOME_TEAM_ID", "elo_home"]].rename(
        columns={"HOME_TEAM_ID": "team_id", "elo_home": "elo"}
    )
    away_elo = df[["GAME_DATE_EST", "VISITOR_TEAM_ID", "elo_away"]].rename(
        columns={"VISITOR_TEAM_ID": "team_id", "elo_away": "elo"}
    )
    long = pd.concat([home_elo, away_elo], ignore_index=True)
    long = long.sort_values(["team_id", "GAME_DATE_EST"]).reset_index(drop=True)

    # Rolling mean of ELO over last 5 games — shift(1) so current game not included
    long["elo_rolling_five"] = (
        long.groupby("team_id")["elo"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    # Merge back
    home_roll = long.rename(columns={
        "team_id": "HOME_TEAM_ID",
        "elo_rolling_five": "elo_rolling_five_home",
    })[["GAME_DATE_EST", "HOME_TEAM_ID", "elo_rolling_five_home"]]

    away_roll = long.rename(columns={
        "team_id": "VISITOR_TEAM_ID",
        "elo_rolling_five": "elo_rolling_five_away",
    })[["GAME_DATE_EST", "VISITOR_TEAM_ID", "elo_rolling_five_away"]]

    df = df.merge(
        home_roll.drop_duplicates(["GAME_DATE_EST", "HOME_TEAM_ID"]),
        on=["GAME_DATE_EST", "HOME_TEAM_ID"], how="left"
    )
    df = df.merge(
        away_roll.drop_duplicates(["GAME_DATE_EST", "VISITOR_TEAM_ID"]),
        on=["GAME_DATE_EST", "VISITOR_TEAM_ID"], how="left"
    )

    return df

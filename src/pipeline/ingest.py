"""
ingest.py
---------
Loads and validates all raw NBA CSV files.
Every loader returns a clean, typed DataFrame ready for feature engineering.
"""

import pandas as pd
from pathlib import Path

# Default data directory relative to this file:  ../../data/
_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_min(val) -> float:
    """
    Parse the MIN field from game_details which may be formatted as
    '31.000000:29'  (minutes:seconds)  or  '31.0'  or  NaN.
    Returns total minutes as a float.
    """
    if pd.isna(val):
        return 0.0
    val = str(val).strip()
    if ":" in val:
        parts = val.split(":")
        try:
            minutes = float(parts[0])
            seconds = float(parts[1]) if len(parts) > 1 else 0.0
            return minutes + seconds / 60.0
        except ValueError:
            return 0.0
    try:
        return float(val)
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_games(path: str = None) -> pd.DataFrame:
    """
    Load games_nba.csv.
    One row per game (wide format: home + away columns side-by-side).

    Key columns:
        GAME_ID, GAME_DATE_EST, SEASON, HOME_TEAM_ID, VISITOR_TEAM_ID,
        PTS_home, PTS_away, FG_PCT_home, FG_PCT_away, ..., HOME_TEAM_WINS
    """
    path = path or _DEFAULT_DATA_DIR / "games_nba.csv"
    df = pd.read_csv(path, parse_dates=["GAME_DATE_EST"])

    required = {
        "GAME_ID", "GAME_DATE_EST", "SEASON",
        "HOME_TEAM_ID", "VISITOR_TEAM_ID",
        "PTS_home", "PTS_away",
        "FG_PCT_home", "FG_PCT_away",
        "FT_PCT_home", "FT_PCT_away",
        "FG3_PCT_home", "FG3_PCT_away",
        "AST_home", "AST_away",
        "REB_home", "REB_away",
        "HOME_TEAM_WINS",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"games_nba.csv is missing columns: {missing}")

    # Drop rows where score is unknown (future/cancelled games)
    df = df.dropna(subset=["PTS_home", "PTS_away", "HOME_TEAM_WINS"])

    # Deduplicate at source — raw CSV has 29 duplicate GAME_IDs
    # Keep first occurrence (rows are identical for the duplicates in this dataset)
    df = df.drop_duplicates(subset=["GAME_ID"], keep="first")

    df = df.sort_values("GAME_DATE_EST").reset_index(drop=True)
    df["HOME_TEAM_WINS"] = df["HOME_TEAM_WINS"].astype(int)
    return df


def load_game_details(path: str = None) -> pd.DataFrame:
    """
    Load game_detai.csv.
    One row per player per game (~750k rows).

    Key columns:
        GAME_ID, TEAM_ID, PLAYER_ID, PLAYER_NAME,
        START_POSITION, COMMENT, MIN,
        FGM, FGA, FG3M, FG3A, FTM, FTA,
        OREB, DREB, REB, AST, STL, BLK, TO, PF, PTS, PLUS_MINUS
    """
    path = path or _DEFAULT_DATA_DIR / "game_detai.csv"
    df = pd.read_csv(path, low_memory=False)

    df["MIN"] = df["MIN"].apply(_parse_min)

    # Fill numeric stat columns with 0 (DNP players have NaN)
    stat_cols = [
        "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
        "OREB", "DREB", "REB", "AST", "STL", "BLK", "TO", "PF", "PTS", "PLUS_MINUS",
    ]
    for col in stat_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Injury/DNP flag: COMMENT is non-null when player did not play
    df["is_dnp"] = (
        df["COMMENT"].notna()
        & (df["COMMENT"].str.strip() != "")
        & (df["MIN"] == 0)
    ).astype(int)

    return df


def load_ranking(path: str = None) -> pd.DataFrame:
    """
    Load ranking.csv.
    Daily snapshot of team standings.

    Key columns:
        TEAM_ID, STANDINGSDATE, W, L, W_PCT, CONFERENCE
    """
    path = path or _DEFAULT_DATA_DIR / "ranking.csv"
    df = pd.read_csv(path, parse_dates=["STANDINGSDATE"])
    df = df.sort_values("STANDINGSDATE").reset_index(drop=True)
    return df


def load_teams(path: str = None) -> pd.DataFrame:
    """
    Load team.csv.
    Static team metadata.

    Key columns:
        TEAM_ID, ABBREVIATION, NICKNAME, CITY, HEADCOACH
    """
    path = path or _DEFAULT_DATA_DIR / "team.csv"
    return pd.read_csv(path)


def load_players(path: str = None) -> pd.DataFrame:
    """
    Load players_train.csv.
    Player-to-team mapping by season.

    Key columns:
        PLAYER_ID, PLAYER_NAME, TEAM_ID, SEASON
    """
    path = path or _DEFAULT_DATA_DIR / "players_train.csv"
    return pd.read_csv(path)


def load_all(data_dir: str = None) -> dict:
    """
    Convenience loader — returns all DataFrames as a dictionary.

    Usage:
        raw = load_all()
        games = raw["games"]
    """
    base = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
    return {
        "games":   load_games(base / "games_nba.csv"),
        "details": load_game_details(base / "game_detai.csv"),
        "ranking": load_ranking(base / "ranking.csv"),
        "teams":   load_teams(base / "team.csv"),
        "players": load_players(base / "players_train.csv"),
    }

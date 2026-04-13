"""
odds.py
-------
Phase 2: Integrate real betting market data into the master dataset.

Two source files are supported:

  1. nba_betting_money_line.csv  — NBA game_id join (2006-2017, Pinnacle Sports)
       Columns: game_id, book_name, team_id, a_team_id, price1, price2
       Layout : team_id = away, a_team_id = home
                price1  = away moneyline, price2 = home moneyline

  2. nba_odds_2007_2024.csv      — date + team-abbrev join (2007-2025)
       Columns: season, date, away, home, spread, moneyline_away, moneyline_home
       Moneyline missing for 2023-partial, 2024, 2025  →  use spread fallback

Strategy (applied in order):
  a) Pinnacle moneyline (2006-2017) — most accurate closing line
  b) nba_odds_2007_2024 moneyline (2018-2022) — fills post-Pinnacle window
  c) nba_odds_2007_2024 spread → implied prob (2023+, no moneyline)
  d) ELO implied_prob_home (2003-2006, any remaining gaps)

Output columns added to master dataset:
  home_implied_prob_close  — market-implied win probability for home team (vig-removed)
  away_implied_prob_close  — same for away team
  home_spread_close        — closing spread (negative = home favored)
  market_elo_diff          — home_implied_prob_close − elo_implied_prob_home
                             (sharp-money signal: market vs model disagreement)
  has_market_odds          — 1 if real odds available, 0 = ELO fallback
"""

import numpy as np
import pandas as pd
from pathlib import Path

_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# ---------------------------------------------------------------------------
# Team abbreviation → NBA TEAM_ID
# Covers all 30 current franchises + common historical abbreviations
# ---------------------------------------------------------------------------
ABBREV_TO_TEAM_ID: dict[str, int] = {
    # Standard NBA abbreviations
    "atl": 1610612737,
    "bos": 1610612738,
    "nop": 1610612740,
    "chi": 1610612741,
    "dal": 1610612742,
    "den": 1610612743,
    "hou": 1610612745,
    "lac": 1610612746,
    "lal": 1610612747,
    "mia": 1610612748,
    "mil": 1610612749,
    "min": 1610612750,
    "bkn": 1610612751,
    "nyk": 1610612752,
    "orl": 1610612753,
    "ind": 1610612754,
    "phi": 1610612755,
    "phx": 1610612756,
    "por": 1610612757,
    "sac": 1610612758,
    "sas": 1610612759,
    "okc": 1610612760,
    "tor": 1610612761,
    "uta": 1610612762,
    "mem": 1610612763,
    "was": 1610612764,
    "det": 1610612765,
    "cha": 1610612766,
    "cle": 1610612739,
    "gsw": 1610612744,
    # Aliases used in nba_odds_2007_2024.csv
    "gs":   1610612744,  # Golden State Warriors
    "sa":   1610612759,  # San Antonio Spurs
    "no":   1610612740,  # New Orleans (Pelicans / Hornets)
    "ny":   1610612752,  # New York Knicks
    "utah": 1610612762,  # Utah Jazz
    "wsh":  1610612764,  # Washington Wizards
    "nj":   1610612751,  # New Jersey Nets (became BKN)
    "van":  1610612763,  # Vancouver Grizzlies (became MEM)
    "sea":  1610612760,  # Seattle SuperSonics (became OKC)
    "noh":  1610612740,  # New Orleans Hornets
    "nok":  1610612740,  # NO/Oklahoma City Hornets
    "noj":  1610612740,  # alt abbreviation
}


# ---------------------------------------------------------------------------
# Moneyline → implied probability
# ---------------------------------------------------------------------------

def _ml_to_raw_prob(ml: float) -> float:
    """
    Convert American moneyline to raw (vig-inclusive) implied probability.
      Negative odds (favourite): |ml| / (|ml| + 100)
      Positive odds (underdog) : 100  / (ml + 100)
    Returns NaN for NaN / zero input.
    """
    if pd.isna(ml) or ml == 0:
        return np.nan
    if ml < 0:
        return abs(ml) / (abs(ml) + 100.0)
    else:
        return 100.0 / (ml + 100.0)


def ml_pair_to_novig_prob(ml_home: float, ml_away: float) -> tuple[float, float]:
    """
    Convert a home/away moneyline pair to vig-removed (true) probabilities.

    Returns (prob_home, prob_away) with prob_home + prob_away == 1.
    Returns (NaN, NaN) if either input is NaN.
    """
    p_home_raw = _ml_to_raw_prob(ml_home)
    p_away_raw = _ml_to_raw_prob(ml_away)

    if pd.isna(p_home_raw) or pd.isna(p_away_raw):
        return np.nan, np.nan

    total = p_home_raw + p_away_raw
    if total <= 0:
        return np.nan, np.nan

    return p_home_raw / total, p_away_raw / total


def spread_to_implied_prob(home_spread_signed: float) -> float:
    """
    Estimate home-team win probability from the SIGNED closing point spread.

    Convention (home perspective):
        home_spread_signed < 0  →  home team is favored (e.g. -6.5)  →  prob > 0.5
        home_spread_signed > 0  →  home team is underdog (e.g. +5)   →  prob < 0.5

    Formula: prob = sigmoid(-home_spread_signed * 0.1412)
    Coefficient 0.1412 calibrated on 19,817 NBA games against vig-removed
    moneyline probabilities (MSE 0.000537).
    """
    if pd.isna(home_spread_signed):
        return np.nan
    return float(1.0 / (1.0 + np.exp(home_spread_signed * 0.1412)))


# ---------------------------------------------------------------------------
# Source 1: Pinnacle moneyline via game_id join
# ---------------------------------------------------------------------------

def _load_pinnacle_odds(data_dir: Path) -> pd.DataFrame:
    """
    Load nba_betting_money_line.csv, filter to Pinnacle Sports, and return
    one row per game with columns:
        game_id, home_ml, away_ml, home_implied_prob, away_implied_prob
    """
    path = data_dir / "nba_betting_money_line.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    df = df[df["book_name"] == "Pinnacle Sports"].copy()

    # Layout: team_id = away, a_team_id = home
    #         price1  = away ML, price2 = home ML
    df = df.rename(columns={
        "game_id":   "game_id",
        "a_team_id": "home_team_id",
        "team_id":   "away_team_id",
        "price2":    "home_ml",
        "price1":    "away_ml",
    })

    # Compute vig-removed probabilities row-wise
    probs = df.apply(
        lambda r: pd.Series(ml_pair_to_novig_prob(r["home_ml"], r["away_ml"]),
                            index=["home_implied_prob", "away_implied_prob"]),
        axis=1,
    )
    df = pd.concat([df[["game_id", "home_team_id", "away_team_id",
                         "home_ml", "away_ml"]], probs], axis=1)

    df = df.dropna(subset=["home_implied_prob"])
    df["source"] = "pinnacle"
    return df


# ---------------------------------------------------------------------------
# Source 2: nba_odds_2007_2024 via date + abbreviation join
# ---------------------------------------------------------------------------

def _load_nba_odds_2007_2024(data_dir: Path) -> pd.DataFrame:
    """
    Load nba_odds_2007_2024.csv and normalise to:
        game_date (datetime), home_abbrev, away_abbrev,
        home_team_id, away_team_id,
        home_ml, away_ml,
        home_spread,
        home_implied_prob, away_implied_prob  (moneyline or spread fallback)
    """
    path = data_dir / "nba_odds_2007_2024.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path, parse_dates=["date"])

    # Normalise abbreviations to TEAM_IDs
    df["home_abbrev"] = df["home"].str.strip().str.lower()
    df["away_abbrev"] = df["away"].str.strip().str.lower()
    df["home_team_id"] = df["home_abbrev"].map(ABBREV_TO_TEAM_ID)
    df["away_team_id"] = df["away_abbrev"].map(ABBREV_TO_TEAM_ID)

    # Drop rows where we couldn't map a team
    unmapped_home = df["home_team_id"].isna()
    unmapped_away = df["away_team_id"].isna()
    if unmapped_home.any() or unmapped_away.any():
        bad_home = df.loc[unmapped_home, "home_abbrev"].unique()
        bad_away = df.loc[unmapped_away, "away_abbrev"].unique()
        import warnings
        warnings.warn(f"Unmapped team abbreviations — home: {bad_home}, away: {bad_away}")
    df = df.dropna(subset=["home_team_id", "away_team_id"])
    df["home_team_id"] = df["home_team_id"].astype(int)
    df["away_team_id"] = df["away_team_id"].astype(int)

    # Rename raw moneyline columns
    df = df.rename(columns={
        "date":           "game_date",
        "moneyline_home": "home_ml",
        "moneyline_away": "away_ml",
    })

    # Compute SIGNED spread from home perspective:
    #   home favored  →  home_spread = -spread  (e.g. home -13)
    #   away favored  →  home_spread = +spread  (e.g. home +5 underdog)
    df["home_spread"] = np.where(
        df["whos_favored"] == "home",
        -df["spread"].abs(),
        +df["spread"].abs(),
    )

    # Compute implied probabilities
    # Where moneyline available: use vig-removed moneyline conversion
    # Where moneyline missing:   use spread fallback with calibrated formula
    rows_with_ml = df["home_ml"].notna() & df["away_ml"].notna()

    probs_ml = df[rows_with_ml].apply(
        lambda r: pd.Series(ml_pair_to_novig_prob(r["home_ml"], r["away_ml"]),
                            index=["home_implied_prob", "away_implied_prob"]),
        axis=1,
    )
    df.loc[rows_with_ml, "home_implied_prob"] = probs_ml["home_implied_prob"].values
    df.loc[rows_with_ml, "away_implied_prob"] = probs_ml["away_implied_prob"].values

    rows_no_ml = ~rows_with_ml
    df.loc[rows_no_ml, "home_implied_prob"] = (
        df.loc[rows_no_ml, "home_spread"].apply(spread_to_implied_prob)
    )
    df.loc[rows_no_ml, "away_implied_prob"] = 1.0 - df.loc[rows_no_ml, "home_implied_prob"]

    df["source"] = rows_with_ml.map({True: "nba_odds_ml", False: "nba_odds_spread"})

    keep = ["game_date", "home_team_id", "away_team_id",
            "home_ml", "away_ml", "home_spread",
            "home_implied_prob", "away_implied_prob", "source"]
    return df[keep].dropna(subset=["home_implied_prob"])


# ---------------------------------------------------------------------------
# Main: build merged odds features
# ---------------------------------------------------------------------------

def build_odds_features(
    games_df: pd.DataFrame,
    data_dir=None,
) -> pd.DataFrame:
    """
    Join all odds sources onto games_df and return a DataFrame with
    one row per game (keyed on GAME_ID) containing:

        home_implied_prob_close   — best available market probability
        away_implied_prob_close
        home_spread_close         — closing spread (home perspective)
        market_elo_diff           — market_prob − elo_prob  (sharp signal)
        has_market_odds           — 1 = real odds, 0 = ELO fallback

    Merge priority (first non-NaN wins):
        1. Pinnacle moneyline (game_id join, 2006-2017)
        2. nba_odds_2007_2024 moneyline (date+team join, 2018-2022)
        3. nba_odds_2007_2024 spread fallback (2023-2024)
        4. ELO implied_prob_home (2003-2005, any remaining gap)
    """
    base = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR

    # ── Load odds sources ─────────────────────────────────────────────────
    pinnacle  = _load_pinnacle_odds(base)
    nba_odds  = _load_nba_odds_2007_2024(base)

    # ── Start with the game index ─────────────────────────────────────────
    result = games_df[["GAME_ID", "GAME_DATE_EST",
                        "HOME_TEAM_ID", "VISITOR_TEAM_ID",
                        "implied_prob_home"]].copy()
    result = result.rename(columns={"GAME_DATE_EST": "_date"})
    result["_date"] = pd.to_datetime(result["_date"])
    result["home_implied_prob_close"] = np.nan
    result["away_implied_prob_close"] = np.nan
    result["home_spread_close"]       = np.nan
    result["_odds_source"]            = "elo_fallback"

    # ── Merge source 1: Pinnacle (game_id) ───────────────────────────────
    if not pinnacle.empty:
        p_merge = pinnacle[["game_id", "home_implied_prob",
                             "away_implied_prob", "source"]].copy()
        p_merge = p_merge.rename(columns={
            "game_id":           "GAME_ID",
            "home_implied_prob": "_p_home",
            "away_implied_prob": "_p_away",
            "source":            "_p_src",
        })
        result = result.merge(p_merge, on="GAME_ID", how="left")
        mask = result["_p_home"].notna() & result["home_implied_prob_close"].isna()
        result.loc[mask, "home_implied_prob_close"] = result.loc[mask, "_p_home"]
        result.loc[mask, "away_implied_prob_close"] = result.loc[mask, "_p_away"]
        result.loc[mask, "_odds_source"]            = result.loc[mask, "_p_src"]
        result = result.drop(columns=["_p_home", "_p_away", "_p_src"])

    # ── Merge source 2 & 3: nba_odds_2007_2024 (date + teams) ───────────
    if not nba_odds.empty:
        nba_odds_keyed = nba_odds.rename(columns={
            "game_date":        "_date",
            "home_team_id":     "HOME_TEAM_ID",
            "away_team_id":     "VISITOR_TEAM_ID",
            "home_implied_prob":"_o_home",
            "away_implied_prob":"_o_away",
            "home_spread":      "_o_spread",
            "source":           "_o_src",
        })
        nba_odds_keyed["_date"] = pd.to_datetime(nba_odds_keyed["_date"])

        # Deduplicate before merging (same game can appear twice with different season offsets)
        nba_odds_keyed = nba_odds_keyed.drop_duplicates(
            subset=["_date", "HOME_TEAM_ID", "VISITOR_TEAM_ID"], keep="first"
        )

        result = result.merge(
            nba_odds_keyed[["_date", "HOME_TEAM_ID", "VISITOR_TEAM_ID",
                             "_o_home", "_o_away", "_o_spread", "_o_src"]],
            on=["_date", "HOME_TEAM_ID", "VISITOR_TEAM_ID"],
            how="left",
        )
        mask = result["_o_home"].notna() & result["home_implied_prob_close"].isna()
        result.loc[mask, "home_implied_prob_close"] = result.loc[mask, "_o_home"]
        result.loc[mask, "away_implied_prob_close"] = result.loc[mask, "_o_away"]
        result.loc[mask, "home_spread_close"]       = result.loc[mask, "_o_spread"]
        result.loc[mask, "_odds_source"]            = result.loc[mask, "_o_src"]

        # Also fill spread where moneyline came from Pinnacle
        spread_missing = result["home_spread_close"].isna() & result["_o_spread"].notna()
        result.loc[spread_missing, "home_spread_close"] = result.loc[spread_missing, "_o_spread"]

        result = result.drop(columns=["_o_home", "_o_away", "_o_spread", "_o_src"])

    # ── Source 4: ELO fallback for remaining games ───────────────────────
    elo_mask = result["home_implied_prob_close"].isna()
    result.loc[elo_mask, "home_implied_prob_close"] = result.loc[elo_mask, "implied_prob_home"]
    result.loc[elo_mask, "away_implied_prob_close"] = 1.0 - result.loc[elo_mask, "implied_prob_home"]
    result.loc[elo_mask, "_odds_source"] = "elo_fallback"

    # ── Derived features ──────────────────────────────────────────────────
    result["market_elo_diff"] = (
        result["home_implied_prob_close"] - result["implied_prob_home"]
    )
    result["has_market_odds"] = (result["_odds_source"] != "elo_fallback").astype(int)

    # ── Return only the columns we want ───────────────────────────────────
    out_cols = [
        "GAME_ID",
        "home_implied_prob_close",
        "away_implied_prob_close",
        "home_spread_close",
        "market_elo_diff",
        "has_market_odds",
    ]
    return result[out_cols].copy()

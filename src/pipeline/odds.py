"""
odds.py
-------
Phase 2: Integrate real betting market data into the master dataset.

Two source files are used:

  1. nba_betting_money_line.csv  — CLOSING lines, Pinnacle Sports (2006-2017)
       Columns: game_id, book_name, team_id, a_team_id, price1, price2
       Layout : team_id = away, a_team_id = home
                price1  = away moneyline, price2 = home moneyline

  2. nba_odds_2007_2024.csv      — OPENING lines (2007-2025)
       Confirmed opening: 63% of moneylines diverge >5 pts from Pinnacle
       closing — above typical inter-book variance, consistent with
       opening-to-closing movement (2007-11-01 GS/Utah: -120 open → -156 close).
       Columns: season, date, away, home, spread, moneyline_away, moneyline_home
       Moneyline missing for 2023-partial, 2024, 2025  →  use spread fallback

Phase 1 sharp-money signal (cross-book disagreement):
  nba_betting_money_line.csv contains closing lines from 10 sportsbooks.
  Pinnacle Sports is the world's sharpest market-maker; soft books
  (5Dimes, Bookmaker, Bovada, etc.) adjust their lines after seeing
  where Pinnacle sets the market. Divergence between Pinnacle and soft
  books represents where sharp money has already pushed the line.

    sharp_signal_home = pinnacle_home_prob − mean(soft_books_home_prob)
    Positive → Pinnacle values home MORE → sharp money on home team
    Negative → Pinnacle values away MORE → sharp money on away team

  Validated: home win rate goes from 43% (signal < -2%) to 66% (signal > +2%)
  Coverage: ~14,800 games (2006-2017). NaN outside this range.

Closing-odds strategy (applied in order):
  a) Pinnacle moneyline (2006-2017) — most accurate closing line
  b) nba_odds_2007_2024 moneyline (2018-2022) — fills post-Pinnacle window
     (opening line used as closing proxy; typically 0-0.5 pt difference)
  c) nba_odds_2007_2024 spread → implied prob (2023+, no moneyline)
  d) ELO implied_prob_home (2003-2006, any remaining gaps)

True line movement (line_movement):
  Available for 2007-2017 where both Pinnacle closing AND nba_odds_2007_2024
  opening exist. Computed as:
    line_movement = pinnacle_close_implied_prob − nba_odds_open_implied_prob
  Positive = market moved toward home (sharp money on home)
  Negative = market moved toward away (sharp money on away)
  NaN for 2018+ (no Pinnacle closing available for comparison)

Output columns added to master dataset:
  home_implied_prob_close  — market-implied win probability for home team (vig-removed)
  away_implied_prob_close  — same for away team
  home_spread_close        — closing spread (negative = home favored)
  market_elo_diff          — home_implied_prob_close − elo_implied_prob_home
  has_market_odds          — 1 if real odds available, 0 = ELO fallback
  sharp_signal_home        — Pinnacle − soft_books implied prob gap (Phase 1)
  book_consensus_std       — std of home_prob across all books (market uncertainty)
  open_implied_prob_home   — opening vig-removed home win probability (nba_odds_2007_2024)
  spread_movement_pts      — Pinnacle_close_spread − nba_odds_open_spread
  line_movement            — close_implied_prob − open_implied_prob (2007-2017 only)
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
    Load nba_betting_money_line.csv and nba_betting_spread.csv (Pinnacle Sports)
    and return one row per game with columns:
        game_id, home_ml, away_ml, home_implied_prob, away_implied_prob,
        home_spread_pinnacle  (signed home spread from closing Pinnacle market)
    """
    ml_path     = data_dir / "nba_betting_money_line.csv"
    spread_path = data_dir / "nba_betting_spread.csv"
    if not ml_path.exists():
        return pd.DataFrame()

    # ── Moneylines ─────────────────────────────────────────────────────────
    ml = pd.read_csv(ml_path)
    ml = ml[ml["book_name"] == "Pinnacle Sports"].copy()

    # Layout: team_id = away, a_team_id = home
    #         price1  = away ML, price2 = home ML
    ml = ml.rename(columns={
        "a_team_id": "home_team_id",
        "team_id":   "away_team_id",
        "price2":    "home_ml",
        "price1":    "away_ml",
    })

    probs = ml.apply(
        lambda r: pd.Series(ml_pair_to_novig_prob(r["home_ml"], r["away_ml"]),
                            index=["home_implied_prob", "away_implied_prob"]),
        axis=1,
    )
    df = pd.concat([ml[["game_id", "home_team_id", "away_team_id",
                         "home_ml", "away_ml"]], probs], axis=1)
    df = df.dropna(subset=["home_implied_prob"])

    # ── Closing spreads ─────────────────────────────────────────────────────
    # Layout: team_id = away, a_team_id = home
    #         spread1 = away-team spread, spread2 = home-team spread (signed)
    if spread_path.exists():
        sp = pd.read_csv(spread_path)
        sp = sp[sp["book_name"] == "Pinnacle Sports"].copy()
        sp = sp.rename(columns={"spread2": "home_spread_pinnacle"})[
            ["game_id", "home_spread_pinnacle"]
        ]
        df = df.merge(sp, on="game_id", how="left")
    else:
        df["home_spread_pinnacle"] = np.nan

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
# Phase 2: True line movement from opening odds data
# ---------------------------------------------------------------------------

def _load_opening_lines(data_dir: Path) -> pd.DataFrame:
    """
    Load nba_opening_lines.csv produced by scripts/fetch_opening_lines.py.

    Expected columns (standardised output of the fetch script):
        game_date      : datetime   — game date
        home_team_id   : int        — NBA TEAM_ID for home team
        away_team_id   : int        — NBA TEAM_ID for away team
        open_ml_home   : float      — opening American moneyline for home team
        open_ml_away   : float      — opening American moneyline for away team
        open_spread_home: float     — opening home point spread (signed, negative=fav)

    Returns standardized open/close columns keyed by date + team pair.
    The caller is responsible for merging this onto GAME_ID and deciding
    precedence vs. other odds sources.
    """
    path = data_dir / "nba_opening_lines.csv"
    if not path.exists():
        return pd.DataFrame(columns=[
            "game_date", "home_team_id", "away_team_id",
            "open_home_prob", "open_spread_home",
            "close_home_prob", "close_spread_home",
        ])

    parse_cols = ["game_date"]
    header = pd.read_csv(path, nrows=0)
    if "ingested_at" in header.columns:
        parse_cols.append("ingested_at")
    opening = pd.read_csv(path, parse_dates=parse_cols)
    opening = opening.dropna(subset=["game_date", "home_team_id", "away_team_id"])
    opening["home_team_id"] = opening["home_team_id"].astype(int)
    opening["away_team_id"] = opening["away_team_id"].astype(int)
    opening["game_date"]    = pd.to_datetime(opening["game_date"])

    # Compute vig-removed opening home implied probability
    if "open_ml_home" in opening.columns and "open_ml_away" in opening.columns:
        opening_probs = opening.apply(
            lambda r: pd.Series(
                ml_pair_to_novig_prob(r["open_ml_home"], r["open_ml_away"]),
                index=["open_home_prob", "open_away_prob"],
            ),
            axis=1,
        )
        opening["open_home_prob"] = opening_probs["open_home_prob"]
    elif "open_spread_home" in opening.columns:
        opening["open_home_prob"] = opening["open_spread_home"].apply(spread_to_implied_prob)
    else:
        return pd.DataFrame(columns=[
            "game_date", "home_team_id", "away_team_id",
            "open_home_prob", "open_spread_home",
            "close_home_prob", "close_spread_home",
        ])

    if "close_ml_home" in opening.columns and "close_ml_away" in opening.columns:
        close_probs = opening.apply(
            lambda r: pd.Series(
                ml_pair_to_novig_prob(r["close_ml_home"], r["close_ml_away"]),
                index=["close_home_prob", "close_away_prob"],
            ),
            axis=1,
        )
        opening["close_home_prob"] = close_probs["close_home_prob"]
    elif "close_spread_home" in opening.columns:
        opening["close_home_prob"] = opening["close_spread_home"].apply(spread_to_implied_prob)
    else:
        opening["close_home_prob"] = np.nan

    if "open_spread_home" not in opening.columns:
        opening["open_spread_home"] = np.nan
    if "close_spread_home" not in opening.columns:
        opening["close_spread_home"] = np.nan

    return opening[
        [
            "game_date",
            "home_team_id",
            "away_team_id",
            "open_home_prob",
            "open_spread_home",
            "close_home_prob",
            "close_spread_home",
            *[c for c in ["source", "source_detail", "ingested_at"] if c in opening.columns],
        ]
    ].copy()


# ---------------------------------------------------------------------------
# Phase 1: Cross-book sharp-money signal
# ---------------------------------------------------------------------------

def _compute_sharp_signal(data_dir: Path) -> pd.DataFrame:
    """
    Compute per-game sharp-money proxy from cross-book moneyline disagreement.

    Pinnacle Sports is the sharpest market-maker in sports betting.  Soft
    books (5Dimes, Bookmaker, Bovada, etc.) react to where Pinnacle prices
    the game; they do NOT lead the market.  When Pinnacle's vig-removed
    implied probability for the home team is HIGHER than the soft-book
    average, it means sharp bettors have pushed that line up — and vice versa.

    Validated on 14,670 NBA games (2006-2017):
        sharp_signal_home < -2% → 43.3 % home win rate
        sharp_signal_home  0–1% → 61.2 % home win rate
        sharp_signal_home >  2% → 66.3 % home win rate
    Pearson correlation with home_win: 0.061  (comparable to ELO-based features)

    Parameters
    ----------
    data_dir : Path to the folder containing nba_betting_money_line.csv

    Returns
    -------
    DataFrame with columns:
        GAME_ID             — matches games_df GAME_ID
        sharp_signal_home   — Pinnacle_prob − soft_avg_prob  (float, ~−0.05 to +0.05)
        book_consensus_std  — std of home_prob across ALL books (market uncertainty)

    Games without Pinnacle data return NaN for both columns.
    """
    path = data_dir / "nba_betting_money_line.csv"
    if not path.exists():
        return pd.DataFrame(columns=["GAME_ID", "sharp_signal_home", "book_consensus_std"])

    df = pd.read_csv(path)

    # Layout: team_id = away, a_team_id = home
    #         price1  = away ML,   price2 = home ML
    # Apply vig-removal row-wise using the existing ml_pair_to_novig_prob helper
    probs = df.apply(
        lambda r: ml_pair_to_novig_prob(r["price2"], r["price1"])[0],
        axis=1,
    )
    df = df.copy()
    df["home_prob"] = probs

    # Drop rows with missing or extreme (likely data-entry error) probabilities
    df = df.dropna(subset=["home_prob"])
    df = df[(df["home_prob"] > 0.05) & (df["home_prob"] < 0.95)]

    SHARP_BOOK = "Pinnacle Sports"

    # ── Pinnacle closing home probability (one row per game) ─────────────
    pinnacle = (
        df[df["book_name"] == SHARP_BOOK][["game_id", "home_prob"]]
        .rename(columns={"home_prob": "pinnacle_home_prob"})
        .drop_duplicates(subset=["game_id"], keep="first")
    )

    # ── Soft-book average home probability ───────────────────────────────
    soft = (
        df[df["book_name"] != SHARP_BOOK]
        .groupby("game_id", as_index=False)
        .agg(soft_avg_home_prob=("home_prob", "mean"))
    )

    # ── Market-wide std (all books including Pinnacle) ───────────────────
    # High std → books disagree → large line movement proxy
    consensus_std = (
        df.groupby("game_id", as_index=False)
        .agg(book_consensus_std=("home_prob", "std"))
    )

    # ── Merge and compute signal ─────────────────────────────────────────
    result = (
        pinnacle
        .merge(soft, on="game_id", how="inner")
        .merge(consensus_std, on="game_id", how="left")
    )

    # sharp_signal_home:
    #   Positive → Pinnacle prices home HIGHER than soft books → sharp on home
    #   Negative → Pinnacle prices home LOWER                  → sharp on away
    result["sharp_signal_home"] = (
        result["pinnacle_home_prob"] - result["soft_avg_home_prob"]
    )

    result = result.rename(columns={"game_id": "GAME_ID"})

    return result[["GAME_ID", "sharp_signal_home", "book_consensus_std"]].copy()


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
    opening_lines = _load_opening_lines(base)

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
    result["market_source"]           = "elo_fallback"
    result["opening_source"]          = np.nan

    # ── Merge source 1: Pinnacle (game_id) ───────────────────────────────
    if not pinnacle.empty:
        p_merge = pinnacle[["game_id", "home_implied_prob",
                             "away_implied_prob", "home_spread_pinnacle",
                             "source"]].copy()
        p_merge = p_merge.rename(columns={
            "game_id":              "GAME_ID",
            "home_implied_prob":    "_p_home",
            "away_implied_prob":    "_p_away",
            "home_spread_pinnacle": "_p_spread",
            "source":               "_p_src",
        })
        result = result.merge(p_merge, on="GAME_ID", how="left")
        mask = result["_p_home"].notna() & result["home_implied_prob_close"].isna()
        result.loc[mask, "home_implied_prob_close"] = result.loc[mask, "_p_home"]
        result.loc[mask, "away_implied_prob_close"] = result.loc[mask, "_p_away"]
        result.loc[mask, "home_spread_close"]       = result.loc[mask, "_p_spread"]
        result.loc[mask, "_odds_source"]            = result.loc[mask, "_p_src"]
        result.loc[mask, "market_source"]           = result.loc[mask, "_p_src"]
        result = result.drop(columns=["_p_home", "_p_away", "_p_spread", "_p_src"])

    # ── Merge source 2 & 3: nba_odds_2007_2024 (date + teams) ───────────
    # nba_odds_2007_2024 contains OPENING lines (confirmed: 63% of moneylines
    # diverge >5 pts from Pinnacle closing — above typical inter-book variance).
    # We store the opening prob for ALL games, then use nba_odds as a fallback
    # CLOSING proxy only for games where Pinnacle closing is unavailable (2018+).
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

        # Store opening-line implied prob for ALL games (used for line_movement later)
        result["open_implied_prob_home"] = result["_o_home"]
        result["_open_spread_nba"]       = result["_o_spread"]

        # Use nba_odds as fallback CLOSING for games where Pinnacle is absent (2018+)
        mask = result["_o_home"].notna() & result["home_implied_prob_close"].isna()
        result.loc[mask, "home_implied_prob_close"] = result.loc[mask, "_o_home"]
        result.loc[mask, "away_implied_prob_close"] = result.loc[mask, "_o_away"]
        result.loc[mask, "home_spread_close"]       = result.loc[mask, "_o_spread"]
        result.loc[mask, "_odds_source"]            = result.loc[mask, "_o_src"]
        result.loc[mask, "market_source"]           = result.loc[mask, "_o_src"]

        # Also fill spread where moneyline came from Pinnacle
        spread_missing = result["home_spread_close"].isna() & result["_o_spread"].notna()
        result.loc[spread_missing, "home_spread_close"] = result.loc[spread_missing, "_o_spread"]

        result = result.drop(columns=["_o_home", "_o_away", "_o_spread", "_o_src"])
    else:
        result["open_implied_prob_home"] = np.nan
        result["_open_spread_nba"]       = np.nan

    # ── Canonical opening-lines file: prefer when available ───────────────
    if not opening_lines.empty:
        opening_keyed = opening_lines.rename(columns={
            "game_date": "_date",
            "home_team_id": "HOME_TEAM_ID",
            "away_team_id": "VISITOR_TEAM_ID",
            "open_home_prob": "_ol_open_home",
            "open_spread_home": "_ol_open_spread",
            "close_home_prob": "_ol_close_home",
            "close_spread_home": "_ol_close_spread",
        })
        opening_keyed["_date"] = pd.to_datetime(opening_keyed["_date"])
        opening_keyed = opening_keyed.drop_duplicates(
            subset=["_date", "HOME_TEAM_ID", "VISITOR_TEAM_ID"],
            keep="first",
        )

        result = result.merge(
            opening_keyed[
                [
                    "_date",
                    "HOME_TEAM_ID",
                    "VISITOR_TEAM_ID",
                    "_ol_open_home",
                    "_ol_open_spread",
                    "_ol_close_home",
                    "_ol_close_spread",
                    *[c for c in ["source", "source_detail", "ingested_at"] if c in opening_keyed.columns],
                ]
            ],
            on=["_date", "HOME_TEAM_ID", "VISITOR_TEAM_ID"],
            how="left",
        )

        result["open_implied_prob_home"] = result["_ol_open_home"].combine_first(result["open_implied_prob_home"])
        result["_open_spread_nba"] = result["_ol_open_spread"].combine_first(result["_open_spread_nba"])
        if "source" in opening_keyed.columns:
            result["opening_source"] = result["source"].combine_first(result["opening_source"])

        close_mask = result["_ol_close_home"].notna() & result["home_implied_prob_close"].isna()
        result.loc[close_mask, "home_implied_prob_close"] = result.loc[close_mask, "_ol_close_home"]
        result.loc[close_mask, "away_implied_prob_close"] = 1.0 - result.loc[close_mask, "_ol_close_home"]
        result.loc[close_mask, "_odds_source"] = "opening_lines_close"
        result.loc[close_mask, "market_source"] = "opening_lines_close"

        close_spread_mask = result["home_spread_close"].isna() & result["_ol_close_spread"].notna()
        result.loc[close_spread_mask, "home_spread_close"] = result.loc[close_spread_mask, "_ol_close_spread"]
    else:
        result["_ol_open_home"] = np.nan
        result["_ol_open_spread"] = np.nan
        result["_ol_close_home"] = np.nan
        result["_ol_close_spread"] = np.nan

    # ── Source 4: ELO fallback for remaining games ───────────────────────
    elo_mask = result["home_implied_prob_close"].isna()
    result.loc[elo_mask, "home_implied_prob_close"] = result.loc[elo_mask, "implied_prob_home"]
    result.loc[elo_mask, "away_implied_prob_close"] = 1.0 - result.loc[elo_mask, "implied_prob_home"]
    result.loc[elo_mask, "_odds_source"] = "elo_fallback"
    result.loc[elo_mask, "market_source"] = "elo_fallback"

    # ── Derived features ──────────────────────────────────────────────────
    result["market_elo_diff"] = (
        result["home_implied_prob_close"] - result["implied_prob_home"]
    )
    result["has_market_odds"] = (result["_odds_source"] != "elo_fallback").astype(int)

    # ── Phase 1: Cross-book sharp signal ─────────────────────────────────
    # Merge Pinnacle-vs-soft-books disagreement onto the game index.
    # Games outside 2006-2017 Pinnacle coverage get NaN — XGBoost handles
    # NaN natively so these rows are still used in training.
    sharp = _compute_sharp_signal(base)
    if not sharp.empty:
        result = result.merge(sharp, on="GAME_ID", how="left")
        n_covered = result["sharp_signal_home"].notna().sum()
        import warnings as _w
        _w.warn(
            f"Sharp signal computed for {n_covered:,} / {len(result):,} games "
            f"(NaN = outside Pinnacle 2006-2017 window)",
            stacklevel=2,
        )
    else:
        result["sharp_signal_home"]  = np.nan
        result["book_consensus_std"] = np.nan

    # ── True line movement: Pinnacle closing vs nba_odds_2007_2024 opening ──
    # line_movement = closing_implied_prob − opening_implied_prob
    # Valid ONLY when closing = Pinnacle Sports (2006-2017).
    # For 2018+ games, both "closing" and "opening" come from nba_odds_2007_2024
    # (the same file) so the difference would be spuriously zero — set NaN instead.
    canonical_lm_mask = result["_ol_close_home"].notna() & result["open_implied_prob_home"].notna()
    pinnacle_close_mask = (
        ~canonical_lm_mask
        & (result["_odds_source"] == "pinnacle")
        & result["open_implied_prob_home"].notna()
    )
    result["line_movement"] = np.where(
        canonical_lm_mask | pinnacle_close_mask,
        result["home_implied_prob_close"] - result["open_implied_prob_home"],
        np.nan,
    )

    # spread_movement_pts: Pinnacle closing spread − nba_odds opening spread
    result["spread_movement_pts"] = np.where(
        (canonical_lm_mask | pinnacle_close_mask)
        & result["home_spread_close"].notna()
        & result["_open_spread_nba"].notna(),
        result["home_spread_close"] - result["_open_spread_nba"],
        np.nan,
    )
    result = result.drop(columns=[
        "_open_spread_nba",
        "_ol_open_home",
        "_ol_open_spread",
        "_ol_close_home",
        "_ol_close_spread",
    ])

    n_lm = int(result["line_movement"].notna().sum())
    import warnings as _w
    _w.warn(
        f"True line_movement populated for {n_lm:,} / {len(result):,} games "
        f"(Pinnacle 2006-2017 close vs nba_odds_2007_2024 open)",
        stacklevel=2,
    )

    # ── Return only the columns we want ───────────────────────────────────
    out_cols = [
        "GAME_ID",
        "home_implied_prob_close",
        "away_implied_prob_close",
        "home_spread_close",
        "market_elo_diff",
        "has_market_odds",
        "sharp_signal_home",
        "book_consensus_std",
        "open_implied_prob_home",
        "spread_movement_pts",
        "line_movement",
        "market_source",
        "opening_source",
    ]
    return result[[c for c in out_cols if c in result.columns]].copy()

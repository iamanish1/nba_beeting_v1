"""
features.py
-----------
All feature engineering for the NBA betting prediction model.

Data leakage prevention strategy
---------------------------------
Every rolling feature is computed on the TEAM GAME LOG — a long-format
table with one row per (team, game).  Before any rolling window,
we apply `.shift(1)` so that the current game's result is NEVER
included in the feature value for that game.

Example:
    last_10_win_rate for game G is the win rate across the 10 games
    BEFORE game G, not including game G.

Rolling window rule of thumb:
    window=5  →  recent form  (last ~1 week)
    window=10 →  medium-term performance
    window=20 →  season baseline
"""

import pandas as pd
import numpy as np

# ============================================================================
# STEP 1 — Build the team game log (long format)
# ============================================================================

def build_team_game_log(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wide-format games into a long-format team game log.
    One row per (team, game).

    Columns returned:
        game_id, game_date, season, team_id, opponent_id, is_home,
        pts_scored, pts_allowed,
        fg_pct, ft_pct, fg3_pct, ast, reb,
        won
    """
    home = games_df[[
        "GAME_ID", "GAME_DATE_EST", "SEASON",
        "HOME_TEAM_ID", "VISITOR_TEAM_ID",
        "PTS_home", "PTS_away",
        "FG_PCT_home", "FT_PCT_home", "FG3_PCT_home",
        "AST_home", "REB_home",
        "HOME_TEAM_WINS",
    ]].copy()
    home.columns = [
        "game_id", "game_date", "season",
        "team_id", "opponent_id",
        "pts_scored", "pts_allowed",
        "fg_pct", "ft_pct", "fg3_pct",
        "ast", "reb",
        "won",
    ]
    home["is_home"] = 1

    away = games_df[[
        "GAME_ID", "GAME_DATE_EST", "SEASON",
        "VISITOR_TEAM_ID", "HOME_TEAM_ID",
        "PTS_away", "PTS_home",
        "FG_PCT_away", "FT_PCT_away", "FG3_PCT_away",
        "AST_away", "REB_away",
        "HOME_TEAM_WINS",
    ]].copy()
    away.columns = [
        "game_id", "game_date", "season",
        "team_id", "opponent_id",
        "pts_scored", "pts_allowed",
        "fg_pct", "ft_pct", "fg3_pct",
        "ast", "reb",
        "won",
    ]
    away["won"]     = 1 - away["won"]   # invert: away team wins when HOME_TEAM_WINS=0
    away["is_home"] = 0

    log = pd.concat([home, away], ignore_index=True)
    log = log.sort_values(["team_id", "game_date"]).reset_index(drop=True)
    return log


# ============================================================================
# STEP 2 — Possession stats from game details
# ============================================================================

def compute_possession_stats(details_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-team per-game possession-based stats from game_details.

    Possessions formula (Oliver):
        poss = FGA - OREB + 0.44 * FTA + TO

    Returns DataFrame with columns:
        game_id, team_id,
        possessions,     – estimated team possessions
        to_total,        – total turnovers
        pie_numerator,   – raw PIE numerator (normalised later)
        injured_count,   – players listed as DNP
        active_players   – players with MIN > 0
    """
    grp = details_df.groupby(["GAME_ID", "TEAM_ID"])

    poss = grp.apply(lambda g: pd.Series({
        "possessions":  max(
            g["FGA"].sum() - g["OREB"].sum() + 0.44 * g["FTA"].sum() + g["TO"].sum(),
            1.0,   # avoid division by zero
        ),
        "to_total":     g["TO"].sum(),
        # PIE numerator per team (Oliver's formula)
        "pie_numerator": (
            g["PTS"].sum()
            + g["FGM"].sum()
            + g["FTM"].sum()
            - g["FGA"].sum()
            - g["FTA"].sum()
            + g["DREB"].sum()
            + 0.5 * g["OREB"].sum()
            + g["AST"].sum()
            + g["STL"].sum()
            + 0.5 * g["BLK"].sum()
            - g["PF"].sum()
            - g["TO"].sum()
        ),
        "injured_count": g["is_dnp"].sum(),
        "active_players": (g["MIN"] > 0).sum(),
    })).reset_index()

    poss.columns = ["game_id", "team_id",
                    "possessions", "to_total", "pie_numerator",
                    "injured_count", "active_players"]
    return poss


def compute_pie(poss_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Team PIE (Player Impact Estimate at team level).

    PIE = team_pie_numerator / game_total_pie_numerator
    where game_total = sum of both teams' pie_numerators for the game.

    PIE > 0.5 means team outperformed opponent in combined box-score impact.
    """
    game_total = (
        poss_stats.groupby("game_id")["pie_numerator"]
        .sum()
        .rename("game_pie_total")
        .reset_index()
    )
    df = poss_stats.merge(game_total, on="game_id")
    df["team_pie"] = df["pie_numerator"] / df["game_pie_total"].clip(lower=1e-9)
    return df.drop(columns=["game_pie_total", "pie_numerator"])


# ============================================================================
# STEP 3 — Star player availability
# ============================================================================

def compute_star_availability(
    details_df: pd.DataFrame,
    games_df: pd.DataFrame,
    star_ppg_threshold: float = 20.0,
) -> pd.DataFrame:
    """
    For each game-team, determine whether their star player(s) were available.

    FIX (Phase 1): Previous logic only marked a star as absent if they appeared
    in game_details with is_dnp=1. But truly injured players who don't travel
    are NEVER inserted into game_details at all — so they were silently assumed
    available (default=1), causing star_available=0 to fire on only 71 games.

    New logic (correct):
        Build the set of players who ACTUALLY PLAYED (MIN > 5) per game-team.
        For each expected star, check if their PLAYER_ID is in that set.
        If NOT in the set → they did not play → star absent.
        This catches both DNP rows AND players completely absent from details.

    Star definition: averaged >= star_ppg_threshold PPG in the PREVIOUS season.
    Using prior-season averages avoids leakage.

    Returns DataFrame:
        game_id, team_id, star_available (1/0), star_count, star_points_lost
    """
    game_seasons = games_df[["GAME_ID", "SEASON"]].copy()
    details = details_df.merge(game_seasons, on="GAME_ID", how="inner")

    # ── Step 1: Identify stars from PRIOR season ──────────────────────────────
    # Only use players who actually played meaningful minutes (MIN > 5)
    # to avoid noise from garbage-time appearances inflating averages.
    player_season_avg = (
        details[details["MIN"] > 5]
        .groupby(["PLAYER_ID", "TEAM_ID", "SEASON"])["PTS"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_pts", "count": "games_played"})
    )

    # Shift season forward by 1: a star in season S is expected in season S+1
    prior_season_stars = player_season_avg[
        (player_season_avg["avg_pts"] >= star_ppg_threshold) &
        (player_season_avg["games_played"] >= 20)   # played enough games to be reliable
    ].copy()
    prior_season_stars["SEASON"] = prior_season_stars["SEASON"] + 1
    prior_season_stars = prior_season_stars.drop_duplicates(
        subset=["PLAYER_ID", "TEAM_ID", "SEASON"]
    )

    # Fallback: if threshold finds no stars, take top 2 per team per season
    if prior_season_stars.empty:
        top2 = (
            player_season_avg
            .sort_values("avg_pts", ascending=False)
            .groupby(["TEAM_ID", "SEASON"])
            .head(2)
            .copy()
        )
        top2["SEASON"] = top2["SEASON"] + 1
        prior_season_stars = top2

    # ── Step 2: Build set of players who ACTUALLY PLAYED per game-team ────────
    # A player "played" if they appear in details with MIN > 5.
    # This is the key fix: we check presence in the played set,
    # not existence of a DNP row.
    players_who_played = (
        details[details["MIN"] > 5]
        .drop_duplicates(subset=["GAME_ID", "TEAM_ID", "PLAYER_ID"])
        .groupby(["GAME_ID", "TEAM_ID"])["PLAYER_ID"]
        .apply(set)
        .reset_index()
        .rename(columns={"PLAYER_ID": "played_set"})
    )

    # ── Step 3: For each expected star, check if they are in the played set ───
    # Cross every expected star with every game in their expected season
    all_games_per_season = (
        details[["GAME_ID", "TEAM_ID", "SEASON"]]
        .drop_duplicates(subset=["GAME_ID", "TEAM_ID"])
    )

    # Join stars to the games their team played in their expected season
    star_game_cross = prior_season_stars.merge(
        all_games_per_season,
        on=["TEAM_ID", "SEASON"],
        how="inner"
    )

    # Join in who actually played
    star_game_cross = star_game_cross.merge(
        players_who_played,
        on=["GAME_ID", "TEAM_ID"],
        how="left"
    )

    # Star played = their PLAYER_ID is in the set of players with MIN > 5
    star_game_cross["star_played"] = star_game_cross.apply(
        lambda r: int(r["PLAYER_ID"] in r["played_set"])
        if isinstance(r["played_set"], set) else 0,
        axis=1
    )

    # Points lost = avg_pts of stars who did NOT play
    star_game_cross["pts_lost"] = (
        star_game_cross["avg_pts"] * (1 - star_game_cross["star_played"])
    )

    # ── Step 4: Aggregate to one row per game-team ────────────────────────────
    star_game_agg = (
        star_game_cross
        .groupby(["GAME_ID", "TEAM_ID"])
        .agg(
            stars_who_played=("star_played", "sum"),
            total_stars=("PLAYER_ID", "count"),
            star_points_lost=("pts_lost", "sum"),
        )
        .reset_index()
    )

    # star_available = 1 if at least one star played, 0 if all stars were absent
    star_game_agg["star_available"] = (
        (star_game_agg["stars_who_played"] >= 1).astype(int)
    )

    star_game_agg = (
        star_game_agg
        .rename(columns={
            "GAME_ID": "game_id",
            "TEAM_ID": "team_id",
            "stars_who_played": "star_count",
        })
        [["game_id", "team_id", "star_available", "star_count", "star_points_lost"]]
        .drop_duplicates(subset=["game_id", "team_id"])
        .reset_index(drop=True)
    )

    return star_game_agg


# ============================================================================
# STEP 4 — Rolling team features (applied to team game log)
# ============================================================================

def _rolling(group: pd.Series, window: int, min_periods: int = 3) -> pd.Series:
    """
    Compute rolling mean WITH shift(1) so current game is excluded.
    This is the single function that enforces our no-leakage policy.
    """
    return group.shift(1).rolling(window, min_periods=min_periods).mean()


def add_rolling_performance(log: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling performance features to the team game log.

    All features are computed BEFORE the current game (shift+roll),
    grouped by team_id, sorted chronologically.

    New columns:
        ppg_5, ppg_10               – points per game rolling avg
        opp_ppg_5, opp_ppg_10       – opponent points allowed
        fg_pct_roll_10              – shooting efficiency
        ast_roll_10, reb_roll_10
        last_5_win_rate             – win rate over last 5 games
        last_10_win_rate            – win rate over last 10 games
        net_rating_roll_10          – rolling avg of (pts_scored - pts_allowed)
    """
    df = log.copy().sort_values(["team_id", "game_date"])

    g = df.groupby("team_id")

    df["net_pts"]          = df["pts_scored"] - df["pts_allowed"]

    df["ppg_5"]            = g["pts_scored"].transform(lambda x: _rolling(x, 5))
    df["ppg_10"]           = g["pts_scored"].transform(lambda x: _rolling(x, 10))
    df["opp_ppg_5"]        = g["pts_allowed"].transform(lambda x: _rolling(x, 5))
    df["opp_ppg_10"]       = g["pts_allowed"].transform(lambda x: _rolling(x, 10))
    df["fg_pct_roll_10"]   = g["fg_pct"].transform(lambda x: _rolling(x, 10))
    df["ft_pct_roll_10"]   = g["ft_pct"].transform(lambda x: _rolling(x, 10))
    df["fg3_pct_roll_10"]  = g["fg3_pct"].transform(lambda x: _rolling(x, 10))
    df["ast_roll_10"]      = g["ast"].transform(lambda x: _rolling(x, 10))
    df["reb_roll_10"]      = g["reb"].transform(lambda x: _rolling(x, 10))
    df["last_5_win_rate"]  = g["won"].transform(lambda x: _rolling(x, 5))
    df["last_10_win_rate"] = g["won"].transform(lambda x: _rolling(x, 10))
    df["net_rating_roll_10"] = g["net_pts"].transform(lambda x: _rolling(x, 10))

    return df


def add_possession_rolling(log: pd.DataFrame,
                            poss_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Merge possession stats into the team game log and compute rolling
    offensive/defensive/pace ratings.

    Offensive Rating  = (rolling pts scored / rolling possessions) * 100
    Defensive Rating  = (rolling pts allowed / rolling opp possessions) * 100
    Pace              = rolling possessions per game
    """
    # Bring in possession data and merge
    poss = poss_stats[["game_id", "team_id", "possessions", "to_total", "team_pie"]].copy()
    df = log.merge(poss, on=["game_id", "team_id"], how="left")

    # Fill missing possession estimates (no details for some old games)
    # Fallback: use NBA average ≈ 96 possessions per game
    df["possessions"] = df["possessions"].fillna(96.0)
    df["to_total"]    = df["to_total"].fillna(df["to_total"].median())
    df["team_pie"]    = df["team_pie"].fillna(0.5)

    df = df.sort_values(["team_id", "game_date"]).reset_index(drop=True)

    team_grp = df.groupby("team_id")

    # Rolling sums for rate computation (shift+roll for no-leakage)
    roll_pts_10   = team_grp["pts_scored"].transform(lambda x: x.shift(1).rolling(10, min_periods=3).sum())
    roll_poss_10  = team_grp["possessions"].transform(lambda x: x.shift(1).rolling(10, min_periods=3).sum())
    roll_allow_10 = team_grp["pts_allowed"].transform(lambda x: x.shift(1).rolling(10, min_periods=3).sum())

    df["offensive_rating"]       = (roll_pts_10   / roll_poss_10.clip(lower=1)) * 100
    df["defensive_rating"]       = (roll_allow_10 / roll_poss_10.clip(lower=1)) * 100
    df["net_rating"]             = df["offensive_rating"] - df["defensive_rating"]
    df["pace_roll_10"]           = team_grp["possessions"].transform(lambda x: _rolling(x, 10))
    df["turnovers_per_game"]     = team_grp["to_total"].transform(lambda x: _rolling(x, 10))
    df["player_impact_estimate"] = team_grp["team_pie"].transform(lambda x: _rolling(x, 10))
    df["shooting_pct_roll_10"]   = (
        df["fg_pct_roll_10"] * 2 + df["ft_pct_roll_10"] * 0.44
    ) / 2.44

    return df


def add_rest_fatigue(log: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rest and fatigue features per team.

    rest_days         – calendar days since last game (capped at 10)
    back_to_back      – 1 if rest_days == 1
    fatigue_load_index – games played in prior 7 days (higher = more fatigued)
    """
    df = log.copy().sort_values(["team_id", "game_date"])

    df["prev_game_date"] = df.groupby("team_id")["game_date"].shift(1)
    df["rest_days"] = (
        (df["game_date"] - df["prev_game_date"]).dt.days
        .clip(upper=10)
        .fillna(7)   # first game of season: assume 7 days rest
        .astype(int)
    )
    df["back_to_back"] = (df["rest_days"] == 1).astype(int)

    # Road back-to-back is significantly worse than home back-to-back:
    # team also travelled overnight, slept in a hotel, adjusted to new arena
    df["back_to_back_road"] = (
        (df["back_to_back"] == 1) & (df["is_home"] == 0)
    ).astype(int)

    # Fatigue: count games played in prior 7 calendar days
    def games_last_n_days(group, n=7):
        dates = group["game_date"].values
        result = np.zeros(len(dates), dtype=int)
        for i, d in enumerate(dates):
            # Count games strictly before current game within n days
            cutoff = d - pd.Timedelta(days=n)
            result[i] = np.sum((dates[:i] > cutoff) & (dates[:i] < d))
        return pd.Series(result, index=group.index)

    df["fatigue_load_index"] = (
        df.groupby("team_id", group_keys=False)
        .apply(games_last_n_days)
    )

    df = df.drop(columns=["prev_game_date"])
    return df


def add_coaching_score(log: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Coaching Adaptability Score.

    Measures how consistent the team's net point differential is over the
    last `window` games.  A lower standard deviation = more consistent =
    better coaching.

    Score = 1 / (1 + rolling_std(net_pts, window))
    Range: (0, 1], higher is better.

    Uses shift(1) for no-leakage compliance.
    """
    df = log.copy().sort_values(["team_id", "game_date"])

    df["coaching_adaptability_score"] = (
        df.groupby("team_id")["net_pts"]
        .transform(
            lambda x: 1.0 / (
                1.0 + x.shift(1).rolling(window, min_periods=5).std().fillna(10.0)
            )
        )
    )
    return df


def add_season_pressure(log: pd.DataFrame) -> pd.DataFrame:
    """
    Late-season playoff pressure flag.

    season_pressure = 1 when ALL of:
        - Game is in the last 20% of that team's season schedule  (March/April)
        - Team's rolling win rate is between 40%–60%  (on the playoff bubble)

    Captures the extra effort and focus bubble teams show when every game
    has direct playoff seeding implications.

    No leakage: win rate computed with shift(1) — current game result excluded.
    """
    df = log.copy().sort_values(["team_id", "season", "game_date"])

    # Game number within season per team (1-indexed)
    df["_game_num"] = df.groupby(["team_id", "season"]).cumcount() + 1
    df["_total_games"] = df.groupby(["team_id", "season"])["game_id"].transform("count")
    df["_season_progress"] = df["_game_num"] / df["_total_games"]

    # Rolling win rate up to (but NOT including) the current game — shift(1)
    df["_rolling_win_rate"] = (
        df.groupby(["team_id", "season"])["won"]
        .transform(lambda x: x.shift(1).expanding(min_periods=5).mean())
    )

    df["season_pressure"] = (
        (df["_season_progress"] >= 0.80) &
        (df["_rolling_win_rate"] >= 0.40) &
        (df["_rolling_win_rate"] <= 0.60)
    ).astype(int)

    # Fill NaN (early season games with < 5 prior games) → 0
    df["season_pressure"] = df["season_pressure"].fillna(0).astype(int)

    df = df.drop(columns=["_game_num", "_total_games", "_season_progress", "_rolling_win_rate"])
    return df


def add_opponent_context(log: pd.DataFrame) -> pd.DataFrame:
    """
    Add opponent's rolling offensive/defensive ratings to each team's row.

    opponent_off_rating_10 – the opponent's offensive rating going into this game
    opponent_def_rating_10 – the opponent's defensive rating going into this game

    This is valid (no leakage) because we are using the OPPONENT'S OWN PAST
    performance, not the outcome of THIS game.
    """
    opp_stats = (
        log[["game_id", "team_id", "offensive_rating", "defensive_rating"]]
        .drop_duplicates(subset=["game_id", "team_id"], keep="first")  # prevent self-join fan-out
        .rename(columns={
            "team_id":          "opponent_id",
            "offensive_rating": "opponent_off_rating_10",
            "defensive_rating": "opponent_def_rating_10",
        })
    )

    df = log.merge(opp_stats, on=["game_id", "opponent_id"], how="left")
    return df


# ============================================================================
# STEP 5 — Inject player-level data into team game log
# ============================================================================

def add_player_features(log: pd.DataFrame,
                         poss_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Merge injury counts and star availability into the team game log.
    """
    player_cols = (
        poss_stats[["game_id", "team_id", "injured_count"]]
        .drop_duplicates(subset=["game_id", "team_id"], keep="first")
        .copy()
    )
    player_cols["player_injury_flag"] = (player_cols["injured_count"] > 0).astype(int)

    df = log.merge(player_cols, on=["game_id", "team_id"], how="left")
    df["injured_count"]     = df["injured_count"].fillna(0).astype(int)
    df["player_injury_flag"]= df["player_injury_flag"].fillna(0).astype(int)
    return df


def add_star_features(log: pd.DataFrame,
                       star_availability: pd.DataFrame) -> pd.DataFrame:
    """
    Merge star player availability into the team game log.
    Includes star_points_lost — quantified scoring impact of missing stars.
    """
    df = log.merge(star_availability, on=["game_id", "team_id"], how="left")
    df["star_available"]    = df["star_available"].fillna(1).astype(int)   # default: available
    df["star_count"]        = df["star_count"].fillna(0).astype(int)
    df["star_points_lost"]  = df["star_points_lost"].fillna(0.0)           # default: no stars missing
    return df


# ============================================================================
# STEP 6 — Ranking-based standings features (game-level)
# ============================================================================

def add_ranking_features(games_df: pd.DataFrame,
                          ranking_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join pre-game standings data from ranking.csv onto each game.

    Uses the ranking snapshot from the day BEFORE each game — no leakage.
    ranking.csv has a daily snapshot for every team, so we use merge_asof
    to find the most recent snapshot strictly before game_date.

    New columns added to games_df:
        home_win_pct_home   – home team's HOME record win% going into this game
        road_win_pct_home   – home team's ROAD record win% going into this game
        home_win_pct_away   – away team's HOME record win% going into this game
        road_win_pct_away   – away team's ROAD record win% going into this game
        games_played_home   – games the home team has played this season
        games_played_away   – games the away team has played this season

    Why home/road split matters:
        A team with 60% overall win rate could be 80% at home / 40% on road.
        The current model cannot distinguish this. Home/road splits are
        strong predictors of performance in the correct context.
    """

    def _parse_record(rec):
        """Parse '25-3' → 0.893, '0-0' → 0.5 (neutral prior)."""
        if pd.isna(rec) or str(rec).strip() == "0-0":
            return 0.5
        try:
            w, l = str(rec).strip().split("-")
            total = int(w) + int(l)
            return int(w) / total if total > 0 else 0.5
        except Exception:
            return 0.5

    ranking = ranking_df.copy()
    ranking["home_win_pct"] = ranking["HOME_RECORD"].apply(_parse_record)
    ranking["road_win_pct"] = ranking["ROAD_RECORD"].apply(_parse_record)
    ranking["games_played"] = ranking["G"]
    ranking = ranking[["TEAM_ID", "STANDINGSDATE", "home_win_pct", "road_win_pct", "games_played"]]
    ranking = ranking.sort_values("STANDINGSDATE").reset_index(drop=True)

    # Prepare games with date as datetime
    games = games_df.copy()
    games["_game_date"] = pd.to_datetime(games["GAME_DATE_EST"])

    def _get_pregame_ranking(team_col, suffix):
        """
        For each game, find the ranking snapshot on the day BEFORE the game
        for the team in team_col. Returns a DataFrame with suffix columns.
        """
        team_games = games[["GAME_ID", "_game_date", team_col]].copy()
        team_games = team_games.rename(columns={team_col: "TEAM_ID"})
        # Subtract 1 day: use snapshot strictly BEFORE game date to avoid leakage.
        # ranking.csv snapshots are updated after games finish each day, so the
        # snapshot for Jan 15 already includes Jan 15 results — using it for a
        # Jan 15 game would leak today's outcomes into the feature.
        team_games["_lookup_date"] = team_games["_game_date"] - pd.Timedelta(days=1)
        team_games = team_games.sort_values("_lookup_date")

        result = pd.merge_asof(
            team_games,
            ranking,
            left_on="_lookup_date",
            right_on="STANDINGSDATE",
            by="TEAM_ID",
            direction="backward",       # most recent snapshot strictly before game date
            tolerance=pd.Timedelta("30 days"),  # safety: ignore if > 30 days stale
        )

        result = result.rename(columns={
            "home_win_pct":  f"home_win_pct_{suffix}",
            "road_win_pct":  f"road_win_pct_{suffix}",
            "games_played":  f"games_played_{suffix}",
        })[[
            "GAME_ID",
            f"home_win_pct_{suffix}",
            f"road_win_pct_{suffix}",
            f"games_played_{suffix}",
        ]]
        # restore original sort order
        result = result.sort_values("GAME_ID").reset_index(drop=True)
        return result

    home_ranking = _get_pregame_ranking("HOME_TEAM_ID",    "home")
    away_ranking = _get_pregame_ranking("VISITOR_TEAM_ID", "away")

    games = games.merge(home_ranking, on="GAME_ID", how="left")
    games = games.merge(away_ranking, on="GAME_ID", how="left")

    # Fill missing (first games of season before any snapshot) with neutral 0.5
    for col in ["home_win_pct_home", "road_win_pct_home",
                "home_win_pct_away", "road_win_pct_away"]:
        games[col] = games[col].fillna(0.5)
    for col in ["games_played_home", "games_played_away"]:
        games[col] = games[col].fillna(0).astype(int)

    games = games.drop(columns=["_game_date"])
    return games


# ============================================================================
# STEP 7 — Pace & style features (game-level, from games_df)
# ============================================================================

def add_game_style_features(games_df: pd.DataFrame,
                              log: pd.DataFrame) -> pd.DataFrame:
    """
    Add pre-game style features from rolling historical team form:
        pace_home, pace_away, pace_difference
        shooting_pct_home, shooting_pct_away, shooting_pct_diff
    """
    home_style = (
        log[log["is_home"] == 1][["game_id", "team_id", "pace_roll_10", "shooting_pct_roll_10"]]
        .drop_duplicates(subset=["game_id", "team_id"], keep="first")
        .rename(columns={
            "game_id": "GAME_ID",
            "team_id": "HOME_TEAM_ID",
            "pace_roll_10": "pace_home",
            "shooting_pct_roll_10": "shooting_pct_home",
        })
    )

    away_style = (
        log[log["is_home"] == 0][["game_id", "team_id", "pace_roll_10", "shooting_pct_roll_10"]]
        .drop_duplicates(subset=["game_id", "team_id"], keep="first")
        .rename(columns={
            "game_id": "GAME_ID",
            "team_id": "VISITOR_TEAM_ID",
            "pace_roll_10": "pace_away",
            "shooting_pct_roll_10": "shooting_pct_away",
        })
    )

    df = games_df.merge(home_style, on=["GAME_ID", "HOME_TEAM_ID"], how="left")
    df = df.merge(away_style, on=["GAME_ID", "VISITOR_TEAM_ID"], how="left")

    df["pace_home"] = df["pace_home"].fillna(96.0)
    df["pace_away"] = df["pace_away"].fillna(96.0)
    df["shooting_pct_home"] = df["shooting_pct_home"].fillna(0.5)
    df["shooting_pct_away"] = df["shooting_pct_away"].fillna(0.5)
    df["pace_difference"] = df["pace_home"] - df["pace_away"]
    df["shooting_pct_diff"] = df["shooting_pct_home"] - df["shooting_pct_away"]

    return df

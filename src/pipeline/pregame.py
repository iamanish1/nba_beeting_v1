from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def build_team_game_index(games_df: pd.DataFrame) -> pd.DataFrame:
    home = games_df[["GAME_ID", "GAME_DATE_EST", "SEASON", "HOME_TEAM_ID", "VISITOR_TEAM_ID"]].copy()
    home = home.rename(
        columns={
            "GAME_ID": "game_id",
            "GAME_DATE_EST": "game_date",
            "SEASON": "season",
            "HOME_TEAM_ID": "team_id",
            "VISITOR_TEAM_ID": "opponent_id",
        }
    )
    home["is_home"] = 1

    away = games_df[["GAME_ID", "GAME_DATE_EST", "SEASON", "VISITOR_TEAM_ID", "HOME_TEAM_ID"]].copy()
    away = away.rename(
        columns={
            "GAME_ID": "game_id",
            "GAME_DATE_EST": "game_date",
            "SEASON": "season",
            "VISITOR_TEAM_ID": "team_id",
            "HOME_TEAM_ID": "opponent_id",
        }
    )
    away["is_home"] = 0

    out = pd.concat([home, away], ignore_index=True)
    out["game_date"] = pd.to_datetime(out["game_date"])
    return out.sort_values(["team_id", "season", "game_date", "game_id"]).reset_index(drop=True)


def _mean_overlap(current: set[int], history_sets: list[set[int]], window: int) -> float:
    if not current:
        return 0.0
    recent = history_sets[-window:]
    if not recent:
        return 0.0
    overlaps = [len(current.intersection(prev)) / 5.0 for prev in recent]
    return float(np.mean(overlaps)) if overlaps else 0.0


def compute_local_lineup_impact(details_df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build leakage-safe local lineup and injury-impact features from historical
    game details. Current-game starters are treated as valid pregame proxies,
    while all expectation and impact calculations come strictly from prior games.
    """
    team_games = build_team_game_index(games_df)

    details = details_df.merge(
        team_games[["game_id", "game_date", "season"]].drop_duplicates("game_id"),
        left_on="GAME_ID",
        right_on="game_id",
        how="inner",
    )
    details["player_id"] = pd.to_numeric(details["PLAYER_ID"], errors="coerce").fillna(-1).astype(int)
    details["team_id"] = pd.to_numeric(details["TEAM_ID"], errors="coerce").fillna(-1).astype(int)
    details["played"] = (details["MIN"] > 0).astype(int)
    details["started"] = (
        details["START_POSITION"].fillna("").astype(str).str.strip().ne("")
        & details["played"].eq(1)
    ).astype(int)

    player_games = details[
        [
            "game_id",
            "game_date",
            "season",
            "team_id",
            "player_id",
            "PLAYER_NAME",
            "MIN",
            "PTS",
            "played",
            "started",
        ]
    ].copy()
    player_games = player_games.sort_values(["team_id", "season", "game_date", "game_id", "player_id"])

    rows: list[dict[str, float | int]] = []
    for (team_id, season), group in player_games.groupby(["team_id", "season"], sort=False):
        group = group.sort_values(["game_date", "game_id", "player_id"])
        state: dict[int, dict[str, float]] = {}
        starter_history: list[set[int]] = []
        previous_starters: set[int] = set()

        for game_id, current in group.groupby("game_id", sort=False):
            starter_set = set(current.loc[current["started"] == 1, "player_id"].tolist())
            active_set = set(current.loc[current["played"] == 1, "player_id"].tolist())
            starters_available = len(starter_set)
            returning_starters = len(starter_set.intersection(previous_starters))
            continuity_3 = _mean_overlap(starter_set, starter_history, window=3)
            continuity_5 = _mean_overlap(starter_set, starter_history, window=5)
            continuity_10 = _mean_overlap(starter_set, starter_history, window=10)

            defaults = {
                "game_id": int(game_id),
                "team_id": int(team_id),
                "starters_available": starters_available,
                "starters_missing": 0,
                "top5_minutes_missing": 0,
                "top3_scorers_missing": 0,
                "starter_minutes_share_lost": 0.0,
                "rotation_minutes_share_lost": 0.0,
                "scoring_share_lost": 0.0,
                "injury_impact_score": 0.0,
                "starter_impact_lost": 0.0,
                "lineup_continuity_3": continuity_3,
                "lineup_continuity_5": continuity_5,
                "lineup_continuity_10": continuity_10,
                "returning_starter_count": returning_starters,
                "expected_starter_stability": 0.0,
            }

            prior_rows = [
                {
                    "player_id": player_id,
                    "prior_games": stats["games"],
                    "prior_starts": stats["starts"],
                    "avg_minutes": stats["minutes_sum"] / stats["games"],
                    "avg_points": stats["points_sum"] / stats["games"],
                }
                for player_id, stats in state.items()
                if stats["games"] >= 3
            ]

            if prior_rows:
                player_summary = pd.DataFrame(prior_rows)
                player_summary["start_rate"] = (
                    player_summary["prior_starts"] / player_summary["prior_games"].clip(lower=1)
                )
                player_summary["expected_score"] = (
                    player_summary["start_rate"] * 100.0
                    + player_summary["avg_minutes"]
                    + player_summary["avg_points"] * 0.5
                )

                expected = player_summary.sort_values(
                    ["expected_score", "avg_minutes", "avg_points"], ascending=False
                ).head(5)
                top_minutes = player_summary.sort_values("avg_minutes", ascending=False).head(5)
                top_scorers = player_summary.sort_values("avg_points", ascending=False).head(3)

                expected_set = set(expected["player_id"].tolist())
                missing_expected = expected[~expected["player_id"].isin(active_set)]
                missing_top_minutes = top_minutes[~top_minutes["player_id"].isin(active_set)]
                missing_top_scorers = top_scorers[~top_scorers["player_id"].isin(active_set)]

                expected_minutes_total = float(expected["avg_minutes"].sum())
                top_minutes_total = float(top_minutes["avg_minutes"].sum())
                top_scorers_total = float(top_scorers["avg_points"].sum())
                current_starters = player_summary[player_summary["player_id"].isin(starter_set)]

                defaults.update(
                    {
                        "starters_missing": int(len(expected_set - active_set)),
                        "top5_minutes_missing": int(len(set(top_minutes["player_id"]) - active_set)),
                        "top3_scorers_missing": int(len(set(top_scorers["player_id"]) - active_set)),
                        "starter_minutes_share_lost": (
                            float(missing_expected["avg_minutes"].sum()) / expected_minutes_total
                            if expected_minutes_total > 0
                            else 0.0
                        ),
                        "rotation_minutes_share_lost": (
                            float(missing_top_minutes["avg_minutes"].sum()) / top_minutes_total
                            if top_minutes_total > 0
                            else 0.0
                        ),
                        "scoring_share_lost": (
                            float(missing_top_scorers["avg_points"].sum()) / top_scorers_total
                            if top_scorers_total > 0
                            else 0.0
                        ),
                        "starter_impact_lost": float(missing_expected["avg_minutes"].sum()),
                        "expected_starter_stability": (
                            float(current_starters["start_rate"].mean()) if not current_starters.empty else 0.0
                        ),
                    }
                )
                defaults["injury_impact_score"] = (
                    0.6 * defaults["rotation_minutes_share_lost"] + 0.4 * defaults["scoring_share_lost"]
                )

            rows.append(defaults)
            starter_history.append(starter_set)
            previous_starters = starter_set

            for player_row in current.itertuples(index=False):
                if int(player_row.played) != 1:
                    continue
                player_state = state.setdefault(
                    int(player_row.player_id),
                    {"games": 0.0, "starts": 0.0, "minutes_sum": 0.0, "points_sum": 0.0},
                )
                player_state["games"] += 1.0
                player_state["starts"] += float(player_row.started)
                player_state["minutes_sum"] += float(player_row.MIN)
                player_state["points_sum"] += float(player_row.PTS)

    return pd.DataFrame(rows)


def _load_canonical_csv(path: Path, required: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=required)
    df = pd.read_csv(path, parse_dates=[c for c in ["game_date", "report_timestamp", "confirmed_at", "game_datetime"] if c in pd.read_csv(path, nrows=0).columns])
    for col in required:
        if col not in df.columns:
            df[col] = np.nan
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    for col in ["team_id", "opponent_team_id", "player_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _load_injury_reports(data_dir: Path) -> pd.DataFrame:
    required = [
        "game_date",
        "team_id",
        "opponent_team_id",
        "player_id",
        "player_name",
        "status",
        "impact_score",
        "report_timestamp",
        "source",
    ]
    path = data_dir / "nba_injury_reports.csv"
    df = _load_canonical_csv(path, required)
    if df.empty:
        return df
    df["status"] = df["status"].fillna("").astype(str).str.strip().str.lower()
    df["impact_score"] = pd.to_numeric(df["impact_score"], errors="coerce")
    return df


def _load_daily_lineups(data_dir: Path) -> pd.DataFrame:
    required = [
        "game_date",
        "team_id",
        "opponent_team_id",
        "player_id",
        "player_name",
        "is_confirmed_starter",
        "confirmed_at",
        "game_datetime",
        "source",
    ]
    path = data_dir / "nba_daily_lineups.csv"
    df = _load_canonical_csv(path, required)
    if df.empty:
        return df
    df["is_confirmed_starter"] = (
        df["is_confirmed_starter"].fillna(0).astype(float).astype(int)
    )
    return df


def build_external_pregame_features(games_df: pd.DataFrame, data_dir: str | Path | None = None) -> pd.DataFrame:
    """
    Load canonical external pregame injury and lineup files, aggregate them to
    team-game features, and align on the same game identity used by the master
    dataset pipeline.
    """
    base = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
    team_games = build_team_game_index(games_df)
    team_games["game_date"] = pd.to_datetime(team_games["game_date"]).dt.normalize()

    injuries = _load_injury_reports(base)
    lineups = _load_daily_lineups(base)

    out = team_games[["game_id", "team_id", "game_date"]].copy()

    if not injuries.empty:
        injury = injuries.copy()
        injury["status_weight"] = injury["status"].map(
            {
                "out": 1.0,
                "inactive": 1.0,
                "doubtful": 0.65,
                "questionable": 0.35,
                "probable": 0.1,
            }
        ).fillna(0.0)
        if injury["impact_score"].notna().any():
            injury["effective_impact"] = injury["status_weight"] * injury["impact_score"].fillna(1.0)
        else:
            injury["effective_impact"] = injury["status_weight"]

        injury_agg = (
            injury.groupby(["game_date", "team_id"])
            .agg(
                questionable_count=("status", lambda s: int((s == "questionable").sum())),
                doubtful_count=("status", lambda s: int((s == "doubtful").sum())),
                out_count=("status", lambda s: int(s.isin(["out", "inactive"]).sum())),
                pregame_injury_impact=("effective_impact", "sum"),
                external_injury_reports_present=("player_name", lambda s: int(s.notna().any())),
            )
            .reset_index()
        )
        out = out.merge(injury_agg, on=["game_date", "team_id"], how="left")
    else:
        out["questionable_count"] = np.nan
        out["doubtful_count"] = np.nan
        out["out_count"] = np.nan
        out["pregame_injury_impact"] = np.nan
        out["external_injury_reports_present"] = 0

    if not lineups.empty:
        lineup = lineups.copy()
        latest_confirmation = lineup.groupby(["game_date", "team_id"])["confirmed_at"].max().rename("confirmed_at_latest")
        lineup_agg = (
            lineup.groupby(["game_date", "team_id"])
            .agg(
                confirmed_starters_available=("is_confirmed_starter", "sum"),
                external_lineups_present=("player_name", lambda s: int(s.notna().any())),
                game_datetime=("game_datetime", "max"),
            )
            .reset_index()
        )
        lineup_agg = lineup_agg.merge(
            latest_confirmation.reset_index(),
            on=["game_date", "team_id"],
            how="left",
        )
        lineup_agg["confirmed_starters_missing"] = np.where(
            lineup_agg["external_lineups_present"] == 1,
            (5 - lineup_agg["confirmed_starters_available"]).clip(lower=0),
            np.nan,
        )
        lineup_agg["lineup_confirmation_lag_hours"] = np.where(
            lineup_agg["game_datetime"].notna() & lineup_agg["confirmed_at_latest"].notna(),
            (
                pd.to_datetime(lineup_agg["game_datetime"]) - pd.to_datetime(lineup_agg["confirmed_at_latest"])
            ).dt.total_seconds() / 3600.0,
            np.nan,
        )
        lineup_agg = lineup_agg.drop(columns=["game_datetime", "confirmed_at_latest"])
        out = out.merge(lineup_agg, on=["game_date", "team_id"], how="left")
    else:
        out["confirmed_starters_available"] = np.nan
        out["confirmed_starters_missing"] = np.nan
        out["lineup_confirmation_lag_hours"] = np.nan
        out["external_lineups_present"] = 0

    numeric_defaults = {
        "questionable_count": 0,
        "doubtful_count": 0,
        "out_count": 0,
        "pregame_injury_impact": 0.0,
        "confirmed_starters_available": 0,
        "confirmed_starters_missing": 0,
    }
    for col, default in numeric_defaults.items():
        out[col] = out[col].fillna(default)

    for col in [
        "external_injury_reports_present",
        "external_lineups_present",
        "questionable_count",
        "doubtful_count",
        "out_count",
        "confirmed_starters_available",
        "confirmed_starters_missing",
    ]:
        out[col] = out[col].fillna(0).astype(int)

    return out.drop(columns=["game_date"]).copy()

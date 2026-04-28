from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pipeline.ingest import load_games
from src.pipeline.elo import compute_elo, compute_elo_rolling_five
from src.pipeline.odds import build_odds_features, _load_nba_odds_2007_2024, _load_opening_lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit market-data coverage and source fallback behavior.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    games = load_games(data_dir / "games_nba.csv")
    games = compute_elo_rolling_five(compute_elo(games))
    odds = build_odds_features(games, data_dir=data_dir)
    audit = games[["GAME_ID", "SEASON", "GAME_DATE_EST", "HOME_TEAM_ID", "VISITOR_TEAM_ID"]].merge(
        odds,
        on="GAME_ID",
        how="left",
    )
    audit = audit.rename(columns={"SEASON": "season"})
    audit["GAME_DATE_EST"] = pd.to_datetime(audit["GAME_DATE_EST"])

    nba_odds = _load_nba_odds_2007_2024(data_dir)
    opening = _load_opening_lines(data_dir)
    source_keys = ["GAME_DATE_EST", "HOME_TEAM_ID", "VISITOR_TEAM_ID"]

    if not nba_odds.empty:
        nba_present = nba_odds.rename(
            columns={
                "game_date": "GAME_DATE_EST",
                "home_team_id": "HOME_TEAM_ID",
                "away_team_id": "VISITOR_TEAM_ID",
            }
        )[["GAME_DATE_EST", "HOME_TEAM_ID", "VISITOR_TEAM_ID", "home_implied_prob", "home_spread"]].copy()
        nba_present["GAME_DATE_EST"] = pd.to_datetime(nba_present["GAME_DATE_EST"])
        nba_present["nba_odds_row_present"] = 1
        nba_present["nba_odds_open_prob_present"] = nba_present["home_implied_prob"].notna().astype(int)
        nba_present["nba_odds_spread_present"] = nba_present["home_spread"].notna().astype(int)
        audit = audit.merge(
            nba_present.drop_duplicates(source_keys),
            on=source_keys,
            how="left",
        )
    else:
        audit["nba_odds_row_present"] = 0
        audit["nba_odds_open_prob_present"] = 0
        audit["nba_odds_spread_present"] = 0

    if not opening.empty:
        opening_present = opening.rename(
            columns={
                "game_date": "GAME_DATE_EST",
                "home_team_id": "HOME_TEAM_ID",
                "away_team_id": "VISITOR_TEAM_ID",
            }
        )[["GAME_DATE_EST", "HOME_TEAM_ID", "VISITOR_TEAM_ID", "open_home_prob", "close_home_prob", "open_spread_home", "close_spread_home"]].copy()
        opening_present["GAME_DATE_EST"] = pd.to_datetime(opening_present["GAME_DATE_EST"])
        opening_present["opening_lines_row_present"] = 1
        opening_present["opening_lines_open_prob_present"] = opening_present["open_home_prob"].notna().astype(int)
        opening_present["opening_lines_close_prob_present"] = opening_present["close_home_prob"].notna().astype(int)
        opening_present["opening_lines_open_spread_present"] = opening_present["open_spread_home"].notna().astype(int)
        opening_present["opening_lines_close_spread_present"] = opening_present["close_spread_home"].notna().astype(int)
        audit = audit.merge(
            opening_present.drop_duplicates(source_keys),
            on=source_keys,
            how="left",
        )
    else:
        audit["opening_lines_row_present"] = 0
        audit["opening_lines_open_prob_present"] = 0
        audit["opening_lines_close_prob_present"] = 0
        audit["opening_lines_open_spread_present"] = 0
        audit["opening_lines_close_spread_present"] = 0

    coverage_cols = [
        "home_spread_close",
        "open_implied_prob_home",
        "spread_movement_pts",
        "line_movement",
    ]
    seasonal_coverage = (
        audit.groupby("season")[coverage_cols]
        .apply(lambda frame: frame.notna().mean())
        .reset_index()
    )
    seasonal_coverage.to_csv(output_dir / "market_feature_coverage_by_season.csv", index=False)

    if "market_source" in audit.columns:
        source_mix = (
            audit.groupby(["season", "market_source"])
            .size()
            .reset_index(name="games")
        )
        source_mix.to_csv(output_dir / "market_source_mix.csv", index=False)

    if "opening_source" in audit.columns:
        opening_mix = (
            audit.groupby(["season", "opening_source"])
            .size()
            .reset_index(name="games")
        )
        opening_mix.to_csv(output_dir / "opening_source_mix.csv", index=False)

    audit["missing_reason"] = "covered"
    audit.loc[audit["market_source"].eq("elo_fallback"), "missing_reason"] = "true_source_absence_or_join_miss"
    audit.loc[
        audit["home_spread_close"].isna() & audit["open_implied_prob_home"].notna(),
        "missing_reason",
    ] = "close_join_gap"
    audit.loc[
        audit["open_implied_prob_home"].isna() & audit["market_source"].ne("elo_fallback"),
        "missing_reason",
    ] = "opening_join_gap"
    audit.loc[
        audit["open_implied_prob_home"].isna()
        & audit["nba_odds_row_present"].fillna(0).eq(1)
        & audit["nba_odds_open_prob_present"].fillna(0).eq(0),
        "missing_reason",
    ] = "opening_source_row_present_field_missing"
    audit.loc[
        audit["home_spread_close"].isna()
        & audit["nba_odds_row_present"].fillna(0).eq(1)
        & audit["nba_odds_spread_present"].fillna(0).eq(0)
        & audit["opening_lines_close_spread_present"].fillna(0).eq(0),
        "missing_reason",
    ] = "close_source_row_present_field_missing"
    audit.loc[
        audit["line_movement"].isna()
        & audit["opening_lines_row_present"].fillna(0).eq(1)
        & (
            audit["opening_lines_open_prob_present"].fillna(0).eq(0)
            | audit["opening_lines_close_prob_present"].fillna(0).eq(0)
        ),
        "missing_reason",
    ] = "no_real_open_close_pair"

    recent_gaps = audit[
        (audit["season"] >= 2018)
        & (
            audit["home_spread_close"].isna()
            | audit["open_implied_prob_home"].isna()
            | audit["line_movement"].isna()
        )
    ].copy()
    recent_gaps.to_csv(output_dir / "recent_market_gaps.csv", index=False)

    reason_summary = (
        recent_gaps.groupby(["season", "missing_reason"])
        .size()
        .reset_index(name="games")
    )
    reason_summary.to_csv(output_dir / "recent_market_gap_reasons.csv", index=False)

    print(f"Saved market audit reports to {output_dir}")


if __name__ == "__main__":
    main()

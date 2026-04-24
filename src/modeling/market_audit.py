from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pipeline.ingest import load_games
from src.pipeline.elo import compute_elo, compute_elo_rolling_five
from src.pipeline.odds import build_odds_features


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

    recent_gaps = audit[
        (audit["season"] >= 2018)
        & (
            audit["home_spread_close"].isna()
            | audit["open_implied_prob_home"].isna()
            | audit["line_movement"].isna()
        )
    ].copy()
    recent_gaps.to_csv(output_dir / "recent_market_gaps.csv", index=False)

    print(f"Saved market audit reports to {output_dir}")


if __name__ == "__main__":
    main()

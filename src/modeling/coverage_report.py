from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.modeling.common import apply_neutral_feature_defaults, load_master_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Seasonal feature coverage audit for master dataset.")
    parser.add_argument("--master-path", default="data/master_dataset.csv")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    master = load_master_dataset(args.master_path)
    raw_master = master.copy()
    master = apply_neutral_feature_defaults(master)

    cols = [
        "home_spread_close",
        "sharp_signal_home",
        "book_consensus_std",
        "spread_movement_pts",
        "open_implied_prob_home",
        "line_movement",
        "h2h_home_win_rate_10",
        "h2h_home_pts_diff_10",
        "injury_impact_score_home",
        "injury_impact_score_away",
        "lineup_continuity_5_home",
        "lineup_continuity_5_away",
        "pregame_injury_impact_home",
        "pregame_injury_impact_away",
        "confirmed_starters_available_home",
        "confirmed_starters_available_away",
    ]
    cols = [c for c in cols if c in raw_master.columns]

    coverage = (
        raw_master.groupby("season")[cols]
        .apply(lambda frame: frame.notna().mean())
        .reset_index()
    )
    coverage.to_csv(output_dir / "seasonal_feature_coverage.csv", index=False)

    h2h_effect = pd.DataFrame(
        {
            "column": ["h2h_home_win_rate_10", "h2h_home_pts_diff_10"],
            "raw_missing": [
                int(raw_master["h2h_home_win_rate_10"].isna().sum()) if "h2h_home_win_rate_10" in raw_master.columns else 0,
                int(raw_master["h2h_home_pts_diff_10"].isna().sum()) if "h2h_home_pts_diff_10" in raw_master.columns else 0,
            ],
            "post_fill_missing": [
                int(master["h2h_home_win_rate_10"].isna().sum()) if "h2h_home_win_rate_10" in master.columns else 0,
                int(master["h2h_home_pts_diff_10"].isna().sum()) if "h2h_home_pts_diff_10" in master.columns else 0,
            ],
        }
    )
    h2h_effect.to_csv(output_dir / "h2h_fill_audit.csv", index=False)

    present_cols = [
        "external_injury_reports_present_home",
        "external_injury_reports_present_away",
        "external_lineups_present_home",
        "external_lineups_present_away",
    ]
    present_cols = [c for c in present_cols if c in master.columns]
    if present_cols:
        presence = (
            master.groupby("season")[present_cols]
            .mean()
            .reset_index()
        )
        presence.to_csv(output_dir / "pregame_feed_presence.csv", index=False)

    print(f"Saved coverage reports to {output_dir}")


if __name__ == "__main__":
    main()

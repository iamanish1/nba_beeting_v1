from __future__ import annotations

import argparse
from io import BytesIO, StringIO
from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT = DATA_DIR / "nba_daily_lineups.csv"
TEAM_PATH = DATA_DIR / "team.csv"


def _load_raw_frame(input_path: str | None, url: str | None) -> pd.DataFrame:
    if input_path:
        path = Path(input_path)
        if path.suffix.lower() == ".json":
            return pd.read_json(path)
        return pd.read_csv(path)
    if url:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        if url.lower().endswith(".json"):
            return pd.read_json(BytesIO(response.content))
        return pd.read_csv(StringIO(response.text))
    raise SystemExit("Provide --input or --url.")


def _team_lookup() -> dict[str, int]:
    teams = pd.read_csv(TEAM_PATH)
    lookup: dict[str, int] = {}
    for row in teams.itertuples(index=False):
        team_id = int(row.TEAM_ID)
        for value in {
            str(getattr(row, "ABBREVIATION", "")),
            str(getattr(row, "CITY", "")),
            str(getattr(row, "NICKNAME", "")),
            f"{getattr(row, 'CITY', '')} {getattr(row, 'NICKNAME', '')}",
        }:
            key = value.lower().strip().replace(" ", "")
            if key:
                lookup[key] = team_id
    lookup.update(
        {
            "gsw": 1610612744,
            "gs": 1610612744,
            "sa": 1610612759,
            "ny": 1610612752,
            "wsh": 1610612764,
            "no": 1610612740,
            "nop": 1610612740,
            "okc": 1610612760,
            "bkn": 1610612751,
        }
    )
    return lookup


def _resolve_col(df: pd.DataFrame, candidates: list[str], required: bool = False) -> str | None:
    lowered = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    if required:
        raise SystemExit(f"Missing required column; tried aliases: {candidates}")
    return None


def _to_bool(series: pd.Series) -> pd.Series:
    return (
        series.fillna(0)
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"1": 1, "true": 1, "yes": 1, "y": 1, "starter": 1, "confirmed": 1, "0": 0, "false": 0, "no": 0, "n": 0})
        .fillna(0)
        .astype(int)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize NBA lineup data into canonical CSV format.")
    parser.add_argument("--input", default=None, help="Local CSV/JSON export to normalize.")
    parser.add_argument("--url", default=None, help="Optional URL returning CSV/JSON.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--source-label", default="external_lineup_feed")
    args = parser.parse_args()

    raw = _load_raw_frame(args.input, args.url)
    lookup = _team_lookup()

    game_date_col = _resolve_col(raw, ["game_date", "date"], required=True)
    team_col = _resolve_col(raw, ["team_id", "team", "team_abbrev", "team_name"], required=True)
    opp_col = _resolve_col(raw, ["opponent_team_id", "opponent", "opponent_team", "opponent_abbrev"])
    player_col = _resolve_col(raw, ["player_name", "player", "name"], required=True)
    player_id_col = _resolve_col(raw, ["player_id", "nba_player_id"])
    starter_col = _resolve_col(raw, ["is_confirmed_starter", "starter", "confirmed", "is_starter"], required=True)
    confirmed_col = _resolve_col(raw, ["confirmed_at", "updated_at", "timestamp"])
    game_dt_col = _resolve_col(raw, ["game_datetime", "commence_time", "tipoff_time"])

    out = pd.DataFrame()
    out["game_date"] = pd.to_datetime(raw[game_date_col]).dt.normalize()

    if team_col.lower().endswith("_id"):
        out["team_id"] = pd.to_numeric(raw[team_col], errors="coerce")
    else:
        out["team_id"] = raw[team_col].astype(str).str.lower().str.replace(" ", "", regex=False).map(lookup)

    if opp_col:
        if opp_col.lower().endswith("_id"):
            out["opponent_team_id"] = pd.to_numeric(raw[opp_col], errors="coerce")
        else:
            out["opponent_team_id"] = raw[opp_col].astype(str).str.lower().str.replace(" ", "", regex=False).map(lookup)
    else:
        out["opponent_team_id"] = pd.NA

    out["player_id"] = pd.to_numeric(raw[player_id_col], errors="coerce") if player_id_col else pd.NA
    out["player_name"] = raw[player_col].astype(str).str.strip()
    out["is_confirmed_starter"] = _to_bool(raw[starter_col])
    out["confirmed_at"] = pd.to_datetime(raw[confirmed_col], errors="coerce") if confirmed_col else pd.NaT
    out["game_datetime"] = pd.to_datetime(raw[game_dt_col], errors="coerce") if game_dt_col else pd.NaT
    out["source"] = args.source_label
    out["source_detail"] = args.input or args.url or args.source_label
    out["ingested_at"] = pd.Timestamp.utcnow().isoformat()

    out = out.dropna(subset=["game_date", "team_id", "player_name"])
    out["team_id"] = out["team_id"].astype(int)
    if out["player_id"].notna().any():
        out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce").astype("Int64")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(f"Saved {len(out):,} lineup rows to {output}")
    print(f"Columns: {out.columns.tolist()}")


if __name__ == "__main__":
    main()

"""
fetch_opening_lines.py
----------------------
Download NBA historical opening + closing odds and save to
data/nba_opening_lines.csv.

Once this file exists the pipeline automatically computes:
    line_movement = closing_implied_prob − opening_implied_prob

Usage:
    python scripts/fetch_opening_lines.py
    python scripts/fetch_opening_lines.py --seasons 2018 2019 2020
    python scripts/fetch_opening_lines.py --source sbro
    python scripts/fetch_opening_lines.py --source oddsapi --api-key YOUR_KEY

Sources supported (tried in order until one succeeds):
    1. sbro  — sportsbookreviewsonline.com  (free, Excel per season)
               Covers 2007-2024, has Open + Close for spread and ML.
    2. oddsapi— The Odds API historical (free tier = last 3 months)
               Full history requires paid plan (~$50/month).

Output columns (standardised, consumed by src/pipeline/odds.py):
    game_date       : YYYY-MM-DD
    home_team_id    : int (NBA TEAM_ID)
    away_team_id    : int (NBA TEAM_ID)
    open_ml_home    : float (American moneyline, e.g. -150)
    open_ml_away    : float (American moneyline, e.g. +130)
    open_spread_home: float (signed home spread, e.g. -4.5)
    close_ml_home   : float
    close_ml_away   : float
    close_spread_home: float
    source          : str   (sbro / oddsapi / etc.)
"""

import argparse
import sys
import warnings
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
OUTPUT_PATH  = DATA_DIR / "nba_opening_lines.csv"
SBRO_DIR     = DATA_DIR / "sbro"

# ── Team name → NBA TEAM_ID (covers SBRO abbreviations + full names) ───────
# SBRO uses city-name or short-name conventions that differ from NBA API.
TEAM_NAME_TO_ID: dict[str, int] = {
    # Standard NBA abbreviations
    "atl": 1610612737, "bos": 1610612738, "nop": 1610612740, "chi": 1610612741,
    "dal": 1610612742, "den": 1610612743, "hou": 1610612745, "lac": 1610612746,
    "lal": 1610612747, "mia": 1610612748, "mil": 1610612749, "min": 1610612750,
    "bkn": 1610612751, "nyk": 1610612752, "orl": 1610612753, "ind": 1610612754,
    "phi": 1610612755, "phx": 1610612756, "por": 1610612757, "sac": 1610612758,
    "sas": 1610612759, "okc": 1610612760, "tor": 1610612761, "uta": 1610612762,
    "mem": 1610612763, "was": 1610612764, "det": 1610612765, "cha": 1610612766,
    "cle": 1610612739, "gsw": 1610612744,
    # SBRO-style team name variations
    "atlanta":      1610612737, "boston":       1610612738,
    "neworleans":   1610612740, "chicago":      1610612741,
    "dallas":       1610612742, "denver":       1610612743,
    "houston":      1610612745, "laclippers":   1610612746,
    "lalakers":     1610612747, "miami":        1610612748,
    "milwaukee":    1610612749, "minnesota":    1610612750,
    "brooklyn":     1610612751, "newyork":      1610612752,
    "orlando":      1610612753, "indiana":      1610612754,
    "philadelphia": 1610612755, "phoenix":      1610612756,
    "portland":     1610612757, "sacramento":   1610612758,
    "sanantonio":   1610612759, "oklacty":      1610612760,
    "oklahomacity": 1610612760, "toronto":      1610612761,
    "utah":         1610612762, "memphis":      1610612763,
    "washington":   1610612764, "detroit":      1610612765,
    "charlotte":    1610612766, "cleveland":    1610612739,
    "goldenstate":  1610612744, "goldenstatewarriors": 1610612744,
    # Historical names
    "nj":    1610612751, "newjersey":   1610612751,  # NJ Nets → Brooklyn
    "noh":   1610612740, "nok":         1610612740,  # NO/OKC Hornets → NOP
    "sea":   1610612760, "seattle":     1610612760,  # Seattle → OKC
    "van":   1610612763, "vancouver":   1610612763,  # Vancouver → Memphis
    "gs":    1610612744,  # GS Warriors shorthand
    "sa":    1610612759,  # San Antonio shorthand
    "no":    1610612740,  # New Orleans shorthand
    "ny":    1610612752,  # New York shorthand
    "wsh":   1610612764,  # Washington shorthand
}


def _name_to_id(name: str) -> int | None:
    """Normalise team name string → NBA TEAM_ID. Returns None if unmapped."""
    key = name.lower().strip().replace(" ", "").replace(".", "")
    return TEAM_NAME_TO_ID.get(key)


# ── SBRO source ─────────────────────────────────────────────────────────────

def _sbro_url(season_start: int) -> str:
    """
    sportsbookreviewsonline.com Excel URL pattern.
    season_start=2022 → nba%20odds%202022-23.xlsx
    """
    season_end_short = str(season_start + 1)[-2:]
    filename = f"nba%20odds%20{season_start}-{season_end_short}.xlsx"
    return (
        f"https://www.sportsbookreviewsonline.com"
        f"/scoresoddsarchives/nba/{filename}"
    )


def _local_sbro_path(season_start: int, sbro_dir: Path = SBRO_DIR) -> Path:
    """Return the expected local SBRO Excel path for a season."""
    season_end_short = str(season_start + 1)[-2:]
    return sbro_dir / f"nba_odds_{season_start}-{season_end_short}.xlsx"


def _parse_sbro_excel(raw_bytes: bytes, season_start: int) -> pd.DataFrame:
    """
    Parse one SBRO Excel file for a single NBA season.

    SBRO NBA Excel format:
        Rows alternate Visitor (V) / Home (H) for each game.
        Columns (positional, headers may be generic):
            0: Date (MMDD or MM/DD/YYYY)
            1: Rot (rotation number — same pair for V/H)
            2: VH  ('V' or 'H')
            3: Team name
            4: 1st Half Open spread (team's side)
            5: 1st Half Total
            6: Full Game Opening Spread
            7: Full Game Closing Spread
            8: Full Game Total Open
            9: Full Game Total Close
            10: Moneyline (closing, or Open/Close in newer files)
           -1: Final Score

    Returns DataFrame with standardised columns or empty DataFrame on parse error.
    """
    try:
        xl = pd.read_excel(BytesIO(raw_bytes), header=0, dtype=str)
    except Exception as e:
        warnings.warn(f"Excel parse failed for {season_start}: {e}")
        return pd.DataFrame()

    xl = xl.reset_index(drop=True)
    ncols = xl.shape[1]

    # Detect column positions by content inspection
    # Column 2 should be 'VH' with values 'V' and 'H'
    vh_col = None
    for c in range(min(5, ncols)):
        vals = xl.iloc[:, c].dropna().str.upper().unique()
        if set(vals) & {"V", "H"}:
            vh_col = c
            break
    if vh_col is None:
        warnings.warn(f"Cannot find VH column in season {season_start} file")
        return pd.DataFrame()

    # Expected column offsets from VH column
    # VH=2: date=0, rot=1, team=3, open_spread=6, close_spread=7, ml=10
    # Handle files where VH might be column 2 or 3
    offset = vh_col - 2  # typically 0

    def _col(n):
        idx = n + offset
        return xl.iloc[:, idx] if 0 <= idx < ncols else pd.Series([np.nan] * len(xl))

    date_raw   = _col(0)
    rot_raw    = _col(1)
    team_raw   = _col(3)
    open_spread= _col(6)
    close_spread=_col(7)
    score_raw  = xl.iloc[:, -1]  # final score = last column

    # Moneyline: some files have open ML at col 10, close at 11
    # Others have only one ML column. Detect by checking unique counts.
    ml_close = _col(10) if ncols > 10 + offset else pd.Series([np.nan]*len(xl))
    ml_open  = _col(11) if ncols > 11 + offset else pd.Series([np.nan]*len(xl))

    df = pd.DataFrame({
        "date_raw":    date_raw.values,
        "rot":         rot_raw.values,
        "vh":          xl.iloc[:, vh_col].values,
        "team_raw":    team_raw.values,
        "open_spread": open_spread.values,
        "close_spread": close_spread.values,
        "ml_col10":    ml_close.values,
        "ml_col11":    ml_open.values,
        "score_raw":   score_raw.values,
    })

    # Filter to data rows (skip header repetitions embedded mid-file)
    df = df[df["vh"].str.upper().isin(["V", "H"])].copy()
    df["vh"] = df["vh"].str.upper()

    # ── Parse date ───────────────────────────────────────────────────────
    def _parse_date(d, yr):
        d = str(d).strip().replace("/", "")
        if len(d) >= 8:
            # MMDDYYYY or YYYYMMDD
            try:
                return pd.to_datetime(d[:8], format="%m%d%Y")
            except Exception:
                try:
                    return pd.to_datetime(d[:8], format="%Y%m%d")
                except Exception:
                    return pd.NaT
        elif len(d) == 4:
            # MMDD — year must be inferred
            mm, dd = int(d[:2]), int(d[2:])
            year = yr if mm >= 10 else yr + 1  # NBA season spans two calendar years
            try:
                return pd.Timestamp(year=year, month=mm, day=dd)
            except Exception:
                return pd.NaT
        return pd.NaT

    df["game_date"] = df["date_raw"].apply(lambda d: _parse_date(d, season_start))
    # Forward-fill dates (SBRO leaves date blank for the H row)
    df["game_date"] = df["game_date"].fillna(method="ffill")

    # ── Pair V/H rows → one row per game ─────────────────────────────────
    visitor = df[df["vh"] == "V"].reset_index(drop=True)
    home    = df[df["vh"] == "H"].reset_index(drop=True)

    if len(visitor) != len(home):
        warnings.warn(
            f"Season {season_start}: unequal V/H rows ({len(visitor)} vs {len(home)}). "
            "Attempting rot-based pairing."
        )
        # Try to pair by rotation number
        visitor = visitor.set_index("rot")
        home    = home.set_index("rot")
        common  = visitor.index.intersection(home.index)
        visitor = visitor.loc[common].reset_index()
        home    = home.loc[common].reset_index()

    games = pd.DataFrame({
        "game_date":      home["game_date"].values,
        "away_team_raw":  visitor["team_raw"].values,
        "home_team_raw":  home["team_raw"].values,
        "open_spread_home": pd.to_numeric(home["open_spread"], errors="coerce").values,
        "close_spread_home": pd.to_numeric(home["close_spread"], errors="coerce").values,
        # ML: if col11 is populated treat col10=open and col11=close
        # otherwise col10 = closing ML only
        "open_ml_home_raw":  pd.to_numeric(home["ml_col11"], errors="coerce").values,
        "close_ml_home_raw": pd.to_numeric(home["ml_col10"], errors="coerce").values,
        "open_ml_away_raw":  pd.to_numeric(visitor["ml_col11"], errors="coerce").values,
        "close_ml_away_raw": pd.to_numeric(visitor["ml_col10"], errors="coerce").values,
    })

    # If col11 is all NaN, col10 is the only ML (closing) → set open = NaN
    if games["open_ml_home_raw"].isna().all():
        games["open_ml_home_raw"] = np.nan
        games["open_ml_away_raw"] = np.nan

    games["home_team_id"] = games["home_team_raw"].apply(_name_to_id)
    games["away_team_id"] = games["away_team_raw"].apply(_name_to_id)

    # Fix spread sign: SBRO records home spread as AWAY perspective sometimes
    # Normalise: open_spread_home < 0 means home favored
    # SBRO convention: the H row's spread is the home team spread (negative=fav)
    # No sign flip needed if already signed correctly.

    games = games.rename(columns={
        "open_ml_home_raw":  "open_ml_home",
        "close_ml_home_raw": "close_ml_home",
        "open_ml_away_raw":  "open_ml_away",
        "close_ml_away_raw": "close_ml_away",
    })

    games["source"] = "sbro"
    n_unmapped = games["home_team_id"].isna().sum()
    if n_unmapped > 0:
        bad = games.loc[games["home_team_id"].isna(), "home_team_raw"].value_counts().head(5)
        warnings.warn(f"Season {season_start}: {n_unmapped} unmapped home teams: {bad.index.tolist()}")

    games = games.dropna(subset=["game_date", "home_team_id", "away_team_id"])
    games["home_team_id"] = games["home_team_id"].astype(int)
    games["away_team_id"] = games["away_team_id"].astype(int)

    return games[[
        "game_date", "home_team_id", "away_team_id",
        "open_ml_home", "open_ml_away", "open_spread_home",
        "close_ml_home", "close_ml_away", "close_spread_home",
        "source",
    ]]


def fetch_sbro(seasons: list[int]) -> pd.DataFrame:
    """
    Download and parse SBRO Excel files for the requested NBA seasons.

    seasons : list of season-start years, e.g. [2018, 2019, 2020]
    """
    all_dfs = []
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; research-bot)"})

    for yr in seasons:
        url = _sbro_url(yr)
        print(f"  Fetching SBRO {yr}-{str(yr+1)[-2:]} ... {url}")
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code != 200:
                print(f"    HTTP {resp.status_code} — skipping season {yr}")
                continue
            parsed = _parse_sbro_excel(resp.content, yr)
            if parsed.empty:
                print(f"    Parse returned 0 rows — skipping season {yr}")
                continue
            all_dfs.append(parsed)
            print(f"    OK — {len(parsed):,} games parsed")
        except requests.RequestException as e:
            print(f"    Request failed: {e} — skipping season {yr}")

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


def fetch_sbro_local_first(seasons: list[int]) -> pd.DataFrame:
    """
    Load local SBRO Excel files from data/sbro/ first, then fall back to
    the legacy remote URL fetcher for any missing seasons.
    """
    all_dfs = []
    missing_remote = []

    for yr in seasons:
        local_path = _local_sbro_path(yr)
        if local_path.exists():
            print(f"  Loading local SBRO {yr}-{str(yr+1)[-2:]} ... {local_path}")
            try:
                parsed = _parse_sbro_excel(local_path.read_bytes(), yr)
            except OSError as e:
                print(f"    Failed to read local file: {e}")
                missing_remote.append(yr)
                continue

            if parsed.empty:
                print("    Local parse returned 0 rows - trying remote URL")
                missing_remote.append(yr)
            else:
                all_dfs.append(parsed)
                print(f"    OK - {len(parsed):,} games parsed from local file")
        else:
            missing_remote.append(yr)

    if missing_remote:
        remote_df = fetch_sbro(missing_remote)
        if not remote_df.empty:
            all_dfs.append(remote_df)

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


# ── The Odds API source ─────────────────────────────────────────────────────

def fetch_odds_api(api_key: str, seasons: list[int]) -> pd.DataFrame:
    """
    Fetch historical NBA odds from The Odds API.
    Free tier: last 3 months of data.
    Paid tier: full history back to 2020.

    api_key: obtain at https://the-odds-api.com
    """
    BASE = "https://api.the-odds-api.com/v4/historical/sports/basketball_nba/odds"
    all_rows = []
    session  = requests.Session()

    # The Odds API returns data per date — we'd need to iterate dates
    # This is a sketch; full implementation requires iterating game dates
    # and paginating. Free tier is very limited for historical data.
    print("  NOTE: The Odds API free tier only covers recent games.")
    print("  For full historical coverage use the SBRO source or a paid plan.")

    params = {
        "apiKey":    api_key,
        "regions":   "us",
        "markets":   "h2h,spreads",
        "oddsFormat":"american",
    }
    try:
        resp = session.get(BASE, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            print(f"  Fetched {len(data)} events from The Odds API")
            # Parse into standardised format
            for event in data:
                game_date = pd.to_datetime(event.get("commence_time", "")).date()
                home_name = event.get("home_team", "")
                away_name = event.get("away_team", "")
                home_id = _name_to_id(home_name)
                away_id = _name_to_id(away_name)
                if home_id is None or away_id is None:
                    continue
                # Extract h2h and spreads from bookmakers
                for bm in event.get("bookmakers", []):
                    for market in bm.get("markets", []):
                        if market["key"] == "h2h":
                            outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
                            all_rows.append({
                                "game_date":    game_date,
                                "home_team_id": home_id,
                                "away_team_id": away_id,
                                "open_ml_home": outcomes.get(home_name),
                                "open_ml_away": outcomes.get(away_name),
                                "open_spread_home": np.nan,
                                "close_ml_home": outcomes.get(home_name),
                                "close_ml_away": outcomes.get(away_name),
                                "close_spread_home": np.nan,
                                "source": f"oddsapi_{bm['key']}",
                            })
                            break
                    break
        else:
            print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"  The Odds API request failed: {e}")

    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch NBA opening + closing odds lines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/fetch_opening_lines.py
  python scripts/fetch_opening_lines.py --seasons 2018 2019 2020
  python scripts/fetch_opening_lines.py --source oddsapi --api-key YOUR_KEY
  python scripts/fetch_opening_lines.py --output data/my_opening_lines.csv
        """,
    )
    parser.add_argument(
        "--seasons", nargs="+", type=int,
        default=list(range(2007, 2025)),
        help="Season start years (default: 2007-2024)",
    )
    parser.add_argument(
        "--source", choices=["sbro", "oddsapi", "auto"],
        default="auto",
        help="Data source to use (default: auto = try sbro first)",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="API key for The Odds API (required for oddsapi source)",
    )
    parser.add_argument(
        "--output", default=str(OUTPUT_PATH),
        help=f"Output CSV path (default: {OUTPUT_PATH})",
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fetching NBA opening lines for seasons: {args.seasons}")
    print(f"Source: {args.source}")
    print()

    df = pd.DataFrame()

    if args.source in ("sbro", "auto"):
        print("Trying sportsbookreviewsonline.com (SBRO)...")
        df = fetch_sbro_local_first(args.seasons)
        if df.empty:
            print("  SBRO returned no data.")
        else:
            print(f"  SBRO: {len(df):,} games fetched across {df['game_date'].nunique()} dates")

    if df.empty and args.source in ("oddsapi", "auto"):
        if args.api_key:
            print("Trying The Odds API...")
            df = fetch_odds_api(args.api_key, args.seasons)
        else:
            print("The Odds API requires --api-key. Skipping.")
            print("Get a free key at https://the-odds-api.com")

    if df.empty:
        print()
        print("No opening lines data fetched.")
        print("Manual options:")
        print("  1. Put local SBRO Excel files in data/sbro/")
        print("     using names like nba_odds_2018-19.xlsx")
        print("  2. Or use The Odds API:")
        print("     python scripts/fetch_opening_lines.py --source oddsapi --api-key YOUR_KEY")
        sys.exit(1)

    # ── Deduplicate ────────────────────────────────────────────────────────
    df["game_date"] = pd.to_datetime(df["game_date"])
    before = len(df)
    df = df.drop_duplicates(
        subset=["game_date", "home_team_id", "away_team_id"], keep="first"
    )
    if before != len(df):
        print(f"Dropped {before - len(df)} duplicate game rows")

    df = df.sort_values("game_date").reset_index(drop=True)

    # ── Save ───────────────────────────────────────────────────────────────
    df.to_csv(output, index=False)
    print()
    print(f"Saved {len(df):,} games to {output}")
    print(f"Columns: {df.columns.tolist()}")
    print()
    print("Coverage by source:")
    print(df["source"].value_counts().to_string())
    print()
    print("Coverage by season (game_date year):")
    df["season_yr"] = df["game_date"].dt.year
    print(df.groupby("season_yr").size().to_string())
    print()
    print(f"Opening ML available: {df['open_ml_home'].notna().mean():.1%} of games")
    print(f"Opening spread available: {df['open_spread_home'].notna().mean():.1%} of games")
    print()
    print("Next step: run the pipeline to rebuild master_dataset.csv")
    print("  python src/pipeline/run.py")


if __name__ == "__main__":
    main()

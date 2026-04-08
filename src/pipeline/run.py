"""
run.py
------
Entry point to build the master dataset from the command line.

Usage:
    python src/pipeline/run.py
    python src/pipeline/run.py --data-dir data/ --output data/master_dataset.csv
"""

import argparse
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pipeline.pipeline import build_master_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Build NBA betting master dataset for XGBoost training."
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to raw CSVs folder (default: <project>/data/)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: <data-dir>/master_dataset.csv)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    output   = args.output

    if output is None:
        base   = Path(data_dir) if data_dir else Path("data")
        output = str(base / "master_dataset.csv")

    master = build_master_dataset(
        data_dir=data_dir,
        output_path=output,
        verbose=not args.quiet,
    )

    print(f"\nDone. Shape: {master.shape}")
    print(f"Columns:\n  " + "\n  ".join(master.columns.tolist()))
    print(f"\nTarget distribution (home_win):")
    print(master["home_win"].value_counts(normalize=True).round(3))
    print(f"\nMissing values per column (top 10):")
    mv = master.isnull().sum().sort_values(ascending=False)
    print(mv[mv > 0].head(10).to_string())


if __name__ == "__main__":
    main()

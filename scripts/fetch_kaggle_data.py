"""
fetch_kaggle_data.py
====================
Download any Kaggle dataset and save it to the data/ folder.
Mirrors the same interactive style as fetch_huggingface_data.py.

USAGE
-----
  python scripts/fetch_kaggle_data.py

You will be asked for:
  1. The Kaggle dataset URL or dataset ID  (e.g. owner/dataset-name)
  2. Which file(s) to keep from the ZIP    (or press Enter to keep all)
  3. Output filename                        (or press Enter to accept suggested)

REQUIREMENTS
------------
  pip install kaggle pandas

KAGGLE API SETUP (one-time)
---------------------------
  1. Go to https://www.kaggle.com  → Account → API → "Create New Token"
  2. This downloads  kaggle.json
  3. Place it at:
       Windows : C:\\Users\\<you>\\.kaggle\\kaggle.json
       Mac/Linux: ~/.kaggle/kaggle.json
  4. The file must contain:  {"username": "...", "key": "..."}

  OR set environment variables:
       KAGGLE_USERNAME=your_username
       KAGGLE_KEY=your_api_key
"""

import os
import re
import sys
import zipfile
import shutil
import tempfile

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
def _check_deps():
    missing = []
    for pkg in ("kaggle", "pandas"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("\n[ERROR] Required packages not installed:")
        for m in missing:
            print(f"        pip install {m}")
        print()
        sys.exit(1)

_check_deps()

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def prompt(message: str, default: str = "") -> str:
    hint = f" [{default}]" if default else ""
    answer = input(f"{message}{hint}: ").strip()
    return answer if answer else default


def extract_dataset_id(raw: str) -> str:
    """Accept a Kaggle URL or bare owner/dataset-name and return the dataset id."""
    raw = raw.strip().rstrip("/")
    # Match https://www.kaggle.com/datasets/<owner>/<name>
    match = re.search(r"kaggle\.com/(?:datasets/)?([^/?#\s]+/[^/?#\s]+)", raw)
    if match:
        return match.group(1)
    # Accept bare ids like  owner/dataset-name
    if re.match(r"^[\w\-\.]+/[\w\-\.]+", raw):
        return raw
    return raw


def authenticate() -> KaggleApi:
    """Authenticate with Kaggle API and return the client."""
    try:
        api = KaggleApi()
        api.authenticate()
        return api
    except Exception as exc:
        print(f"\n[ERROR] Kaggle authentication failed: {exc}")
        print("""
How to fix:
  1. Go to https://www.kaggle.com → Account → API → "Create New Token"
  2. Place the downloaded kaggle.json at:
       Windows : C:\\Users\\<you>\\.kaggle\\kaggle.json
       Mac/Linux: ~/.kaggle/kaggle.json
  3. File contents must be: {"username": "...", "key": "..."}
""")
        sys.exit(1)


def list_dataset_files(api: KaggleApi, dataset_id: str) -> list:
    """Return list of file names in a Kaggle dataset."""
    try:
        files = api.dataset_list_files(dataset_id).files
        return [f.name for f in files]
    except Exception as exc:
        print(f"[ERROR] Could not list dataset files: {exc}")
        return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Kaggle Dataset  →  CSV  downloader")
    print("=" * 60)

    # ── 1. Authenticate ───────────────────────────────────────────────────
    print("\nAuthenticating with Kaggle API …")
    api = authenticate()
    print("  Authentication: OK")

    # ── 2. Dataset ID / URL ───────────────────────────────────────────────
    print("""
Recommended datasets for NBA betting odds (Phase 2):

  [1] cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024
      → Oct 2007 – Jun 2025 | moneyline + spread + totals | 633 KB
      → BEST SINGLE SOURCE for closing lines 2007-2024

  [2] ehallmar/nba-historical-stats-and-betting-data
      → Large dataset with stats + odds | 38 MB
      → Good backup / cross-validation source

  [3] christophertreasure/nba-odds-data
      → 2008-2023 | moneyline + spread + totals | 550 KB
      → Secondary source for cross-checking

  [Enter custom] Paste any Kaggle dataset URL or owner/dataset-name
""")

    raw_input = prompt(
        "Enter number [1-3] or paste a Kaggle URL / dataset ID\n>",
        default="1"
    )

    # Resolve shortcut numbers
    shortcuts = {
        "1": "cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024",
        "2": "ehallmar/nba-historical-stats-and-betting-data",
        "3": "christophertreasure/nba-odds-data",
    }
    if raw_input in shortcuts:
        dataset_id = shortcuts[raw_input]
    else:
        dataset_id = extract_dataset_id(raw_input)

    print(f"\n  Dataset ID : {dataset_id}")

    # ── 3. List files in dataset ──────────────────────────────────────────
    print("\nFetching file list …")
    files = list_dataset_files(api, dataset_id)

    if files:
        print(f"\nFiles in dataset ({len(files)} total):")
        for i, f in enumerate(files, 1):
            print(f"  {i}. {f}")

        file_choice = prompt(
            "\nEnter file number to download, comma-separated numbers, or press Enter for ALL",
            default="all"
        )

        if file_choice.lower() in ("all", ""):
            selected_files = files
        else:
            indices = [x.strip() for x in file_choice.split(",")]
            selected_files = []
            for idx in indices:
                if idx.isdigit():
                    i = int(idx) - 1
                    if 0 <= i < len(files):
                        selected_files.append(files[i])
                    else:
                        print(f"[WARN] Index {idx} out of range, skipping")
                else:
                    # Treat as filename directly
                    selected_files.append(idx)
    else:
        print("  (Could not list files — will download entire dataset)")
        selected_files = []

    # ── 4. Download ────────────────────────────────────────────────────────
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="kaggle_dl_")

    print(f"\nDownloading dataset to temporary folder …")
    try:
        api.dataset_download_files(
            dataset_id,
            path=tmp_dir,
            unzip=True,
            quiet=False,
        )
    except Exception as exc:
        print(f"[ERROR] Download failed: {exc}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        sys.exit(1)

    # ── 5. Find downloaded files ───────────────────────────────────────────
    downloaded = []
    for root, dirs, fnames in os.walk(tmp_dir):
        # Skip __MACOSX and hidden folders
        dirs[:] = [d for d in dirs if not d.startswith("__") and not d.startswith(".")]
        for fname in fnames:
            if not fname.startswith("."):
                downloaded.append(os.path.join(root, fname))

    if not downloaded:
        print("[ERROR] No files found after download.")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        sys.exit(1)

    print(f"\nDownloaded {len(downloaded)} file(s):")
    for f in downloaded:
        size_kb = os.path.getsize(f) / 1024
        print(f"  {os.path.basename(f)}  ({size_kb:.1f} KB)")

    # ── 6. Filter to selected files if specified ───────────────────────────
    if selected_files:
        filtered = [
            f for f in downloaded
            if any(os.path.basename(f) == sf or sf in os.path.basename(f)
                   for sf in selected_files)
        ]
        if filtered:
            downloaded = filtered

    # ── 7. Process each file ───────────────────────────────────────────────
    saved_files = []
    for src_path in downloaded:
        fname = os.path.basename(src_path)
        ext   = os.path.splitext(fname)[1].lower()

        if ext in (".csv", ".xlsx", ".xls", ".json", ".parquet"):
            # Suggest output name
            safe_id   = re.sub(r"[^\w\-]", "_", dataset_id.split("/")[-1])
            suggested = f"{safe_id}_{fname}" if len(downloaded) > 1 else f"{safe_id}.csv"
            suggested = re.sub(r"[^\w\-\.]", "_", suggested)

            out_name = prompt(f"\nSave '{fname}' as", default=suggested)
            if not out_name.endswith(".csv"):
                out_name = os.path.splitext(out_name)[0] + ".csv"

            out_path = os.path.join(DATA_DIR, out_name)

            # Convert to CSV if needed
            try:
                if ext == ".csv":
                    shutil.copy2(src_path, out_path)
                elif ext in (".xlsx", ".xls"):
                    print(f"  Converting Excel → CSV …")
                    df = pd.read_excel(src_path)
                    df.to_csv(out_path, index=False)
                elif ext == ".json":
                    print(f"  Converting JSON → CSV …")
                    df = pd.read_json(src_path)
                    df.to_csv(out_path, index=False)
                elif ext == ".parquet":
                    print(f"  Converting Parquet → CSV …")
                    df = pd.read_parquet(src_path)
                    df.to_csv(out_path, index=False)

                # Show preview
                df_preview = pd.read_csv(out_path, nrows=3)
                rows_total = sum(1 for _ in open(out_path, encoding="utf-8")) - 1
                print(f"\n  Saved: {out_path}")
                print(f"  Rows  : {rows_total:,}")
                print(f"  Cols  : {len(df_preview.columns)}")
                print(f"  Columns: {list(df_preview.columns)}")
                print(f"\n  Sample (first 3 rows):")
                print(df_preview.to_string(index=False))
                saved_files.append(out_path)

            except Exception as exc:
                print(f"[ERROR] Could not process {fname}: {exc}")

        else:
            print(f"  Skipping non-tabular file: {fname} ({ext})")

    # ── 8. Cleanup ─────────────────────────────────────────────────────────
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── 9. Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DOWNLOAD COMPLETE")
    print("=" * 60)
    if saved_files:
        print(f"\n  {len(saved_files)} file(s) saved to data/:")
        for f in saved_files:
            print(f"    {os.path.basename(f)}")
        print("""
  Next steps:
    1. Run this script again to download additional datasets
    2. Once all odds CSVs are in data/, the pipeline will
       merge them automatically when you run:
         python -c "from src.pipeline.pipeline import build_master_dataset; \\
                    build_master_dataset(output_path='data/master_dataset.csv')"
""")
    else:
        print("\n  No files were saved.")

    print("Done!\n")


if __name__ == "__main__":
    main()

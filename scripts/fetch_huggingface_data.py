"""
fetch_huggingface_data.py
=========================
Download any Hugging Face dataset table and save it as a CSV in the data/ folder.

USAGE
-----
  python scripts/fetch_huggingface_data.py

You will be asked for:
  1. The Hugging Face dataset URL  (e.g. https://huggingface.co/datasets/some-user/some-dataset)
     — OR just the dataset ID      (e.g. some-user/some-dataset)
  2. The config name (subset)  — press Enter to use the default
  3. The split name             — press Enter to use "train"
  4. The output filename        — press Enter to accept the suggested name

REQUIREMENTS
------------
  pip install datasets pandas
"""

import re
import sys
import os

# ---------------------------------------------------------------------------
# Dependency check — give a friendly message before crashing
# ---------------------------------------------------------------------------
def _check_deps():
    missing = []
    for pkg in ("datasets", "pandas"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("\n[ERROR] The following packages are required but not installed:")
        for m in missing:
            print(f"        pip install {m}")
        print("\nInstall them and run the script again.\n")
        sys.exit(1)

_check_deps()

# ---------------------------------------------------------------------------
# Imports (safe after dependency check)
# ---------------------------------------------------------------------------
import pandas as pd
from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def extract_dataset_id(raw: str) -> str:
    """Accept a full HF URL or a bare dataset-id and return the dataset id."""
    raw = raw.strip().rstrip("/")
    # Match  https://huggingface.co/datasets/<owner>/<name>
    match = re.search(r"huggingface\.co/datasets/([^/?#\s]+/[^/?#\s]+)", raw)
    if match:
        return match.group(1)
    # Accept bare ids like  owner/dataset  or  owner/dataset:config
    if re.match(r"^[\w\-\.]+/[\w\-\.]+", raw):
        return raw
    return raw  # return as-is and let HF raise a clear error


def prompt(message: str, default: str = "") -> str:
    """Prompt the user and return the answer (or the default on empty input)."""
    hint = f" [{default}]" if default else ""
    answer = input(f"{message}{hint}: ").strip()
    return answer if answer else default


def list_options(label: str, options: list) -> None:
    print(f"\nAvailable {label}:")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Hugging Face Dataset  →  CSV  downloader")
    print("=" * 60)

    # ── 1. Dataset ID / URL ────────────────────────────────────────────────
    raw_input = prompt("\nPaste the Hugging Face dataset URL or dataset ID\n  e.g. https://huggingface.co/datasets/owner/dataset\n  or   owner/dataset\n>")
    if not raw_input:
        print("[ERROR] No dataset provided. Exiting.")
        sys.exit(1)

    dataset_id = extract_dataset_id(raw_input)
    print(f"\n  Dataset ID : {dataset_id}")

    # ── 2. Config (subset) ─────────────────────────────────────────────────
    try:
        configs = get_dataset_config_names(dataset_id)
    except Exception:
        configs = []

    config_name = None
    if configs:
        list_options("configs (subsets)", configs)
        raw_config = prompt(f"\nEnter config name or number", default=configs[0])
        # Allow user to type the list number instead of the full name
        if raw_config.isdigit():
            idx = int(raw_config) - 1
            if 0 <= idx < len(configs):
                config_name = configs[idx]
            else:
                print(f"[ERROR] Number {raw_config} is out of range. Pick 1–{len(configs)}.")
                sys.exit(1)
        else:
            config_name = raw_config
    else:
        print("  (No configs found — using default)")

    # ── 3. Split ───────────────────────────────────────────────────────────
    try:
        splits = get_dataset_split_names(dataset_id, config_name=config_name)
    except Exception:
        splits = ["train"]

    if splits:
        list_options("splits", splits)
        raw_split = prompt("\nEnter split name or number", default=splits[0])
        if raw_split.isdigit():
            idx = int(raw_split) - 1
            if 0 <= idx < len(splits):
                split_name = splits[idx]
            else:
                print(f"[ERROR] Number {raw_split} is out of range. Pick 1–{len(splits)}.")
                sys.exit(1)
        else:
            split_name = raw_split
    else:
        split_name = prompt("\nEnter split name", default="train")

    # ── 4. Load dataset ────────────────────────────────────────────────────
    print(f"\nLoading  '{dataset_id}'  config='{config_name}'  split='{split_name}' …")
    print("(This may take a moment for large datasets)\n")

    try:
        if config_name:
            dataset = load_dataset(dataset_id, config_name, split=split_name)
        else:
            dataset = load_dataset(dataset_id, split=split_name)
    except Exception as exc:
        print(f"[ERROR] Failed to load dataset:\n  {exc}")
        sys.exit(1)

    # ── 5. Convert to DataFrame ────────────────────────────────────────────
    print(f"  Rows loaded : {len(dataset):,}")
    print(f"  Columns     : {', '.join(dataset.column_names)}\n")

    df = dataset.to_pandas()

    # ── 6. Output filename ─────────────────────────────────────────────────
    safe_id   = re.sub(r"[^\w\-]", "_", dataset_id)
    safe_cfg  = f"_{re.sub(r'[^\w]', '_', config_name)}" if config_name else ""
    safe_spl  = re.sub(r"[^\w\-]", "_", split_name)
    suggested = f"{safe_id}{safe_cfg}_{safe_spl}.csv"

    output_filename = prompt(f"Output filename", default=suggested)
    if not output_filename.endswith(".csv"):
        output_filename += ".csv"

    output_path = os.path.join(DATA_DIR, output_filename)

    # ── 7. Save ────────────────────────────────────────────────────────────
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n  Saved {len(df):,} rows  x  {len(df.columns)} columns")
    print(f"  Location : {output_path}")
    print("\nDone!\n")


if __name__ == "__main__":
    main()

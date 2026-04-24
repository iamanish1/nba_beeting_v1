import pandas as pd
import argparse
import os
import sys
import numpy as np

def sample_master_dataset(input_path, output_path, fraction, random_state=42):
    # Read the master dataset
    df = pd.read_csv(input_path)
    # Sample the specified fraction
    sample_df = df.sample(frac=fraction, random_state=random_state)
    # Save to CSV
    sample_df.to_csv(output_path, index=False)
    print(f"Sampled {len(sample_df)} rows out of {len(df)} (fraction={fraction}) to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample a fraction of the master dataset and save to CSV.")
    parser.add_argument('--input', type=str, default="data/master_dataset.csv", help="Path to the master dataset CSV file.")
    parser.add_argument('--output', type=str, default="data/master_dataset_sample.csv", help="Path to save the sampled CSV file.")
    parser.add_argument('--fraction', type=float, default=0.1, help="Fraction of data to sample (e.g., 0.1 for 10%).")
    parser.add_argument('--random_state', type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    if not (0 < args.fraction <= 1):
        print("Error: --fraction must be between 0 and 1.")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        sys.exit(1)

    sample_master_dataset(args.input, args.output, args.fraction, args.random_state)

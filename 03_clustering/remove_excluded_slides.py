#!/usr/bin/env python3
"""Remove excluded slides (e.g. a withdrawn institution) from a subsampled-embedding parquet
(from notebook remove_slides_kmeans.ipynb). The exclusion list is a CSV with a `slides` column.
"""

import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True, help="e.g. Virchow_subsampled_1M.parquet")
    parser.add_argument("--remove_list", required=True, help="CSV with a 'slides' column")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    remove = set(pd.read_csv(args.remove_list)["slides"])
    kept = df[~df["patient"].isin(remove)]
    kept.to_parquet(args.output)
    print(f"Kept {len(kept):,} of {len(df):,} tiles")


if __name__ == "__main__":
    main()

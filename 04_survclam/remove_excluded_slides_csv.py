#!/usr/bin/env python3
"""Remove excluded slides from every SurvCLAM dataset CSV in a folder (from notebook
remove_slides.ipynb). Filtered copies are written to a separate output folder.
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="folder of dataset CSVs")
    parser.add_argument("--remove_list", required=True, help="CSV with a 'slides' column")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--slide_col", default="slide_id")
    args = parser.parse_args()

    remove = set(pd.read_csv(args.remove_list)["slides"])
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(args.input_dir).glob("*.csv")):
        df = pd.read_csv(path)
        if args.slide_col not in df.columns:
            print(f"Skipped {path.name}: no '{args.slide_col}' column")
            continue
        kept = df[~df[args.slide_col].isin(remove)]
        kept.to_csv(out / path.name, index=False)
        print(f"{path.name}: kept {len(kept):,} of {len(df):,} rows")


if __name__ == "__main__":
    main()

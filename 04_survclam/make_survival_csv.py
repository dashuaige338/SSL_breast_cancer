#!/usr/bin/env python3
"""Build the SurvCLAM dataset CSV (from notebook generate_CLAM_csvs.ipynb).

Takes a slide list CSV (column `slide_id`, one row per slide with extracted features) and the
clinical file, and writes case_id, slide_id, time, event for distant recurrence:
time = fup_rec_dec24 (years), event = rec_meta_dec24.
"""

import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slides", required=True, help="CSV with a 'slide_id' column")
    parser.add_argument("--clinical", required=True, help="clinical CSV with 'Primary_scanned_section'")
    parser.add_argument("--output", required=True, help="e.g. dataset_csv/poet_breast_recurrence_virchow_20x.csv")
    parser.add_argument("--time_col", default="fup_rec_dec24")
    parser.add_argument("--event_col", default="rec_meta_dec24")
    args = parser.parse_args()

    meta = pd.read_csv(args.clinical).set_index("Primary_scanned_section")
    slides = pd.read_csv(args.slides)["slide_id"].tolist()
    out = pd.DataFrame({"case_id": slides, "slide_id": slides,
                        "time": meta.loc[slides, args.time_col].to_numpy(),
                        "event": meta.loc[slides, args.event_col].to_numpy()})
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out):,} slides to {args.output}")


if __name__ == "__main__":
    main()

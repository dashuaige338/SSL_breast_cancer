#!/usr/bin/env python3
"""Leave-one-institution-out splits for SurvCLAM (from notebook CLAM_Surv_institution_fold.ipynb).

Each institution with at least one distant-recurrence event, and not listed in --exclude_file,
becomes one test fold (28 folds in the study). For each fold, all other patients are split
80/20 into training and validation (train_test_split, random_state=42); the held-out
institution's patients form the test set. Files: <output_dir>/splits_<i>.csv with columns
train, val, test (the format read by main.py / eval.py).

--exclude_file: one institution name per line (institutions not used as test folds; in the
study, institutions with few events and the institution that withdrew consent). Institution
names are not stored in this repository.
Note: the original notebook ordered folds by Python set iteration, so its fold numbering is
arbitrary; here folds are numbered in sorted order. The train/val/test content of each
institution's fold is the same.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clinical", required=True, help="clinical CSV with 'Primary_scanned_section'")
    parser.add_argument("--hospitals", required=True, help="CSV with columns 'slides' and 'Hospital'")
    parser.add_argument("--exclude_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--event_col", default="rec_meta_dec24")
    parser.add_argument("--time_col", default="fup_rec_dec24")
    args = parser.parse_args()

    meta = pd.read_csv(args.clinical)
    hospital_of = dict(pd.read_csv(args.hospitals)[["slides", "Hospital"]].itertuples(index=False))
    data = pd.DataFrame({"slides": meta["Primary_scanned_section"],
                         "event": meta[args.event_col], "time": meta[args.time_col]})
    data["hospital"] = data.slides.map(hospital_of)
    excluded = {line.strip() for line in Path(args.exclude_file).read_text().splitlines() if line.strip()}
    folds = sorted(set(data.loc[data.event == 1, "hospital"]) - excluded)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for idx, hospital in enumerate(folds):
        test = data.loc[data.hospital == hospital, "slides"].reset_index(drop=True)
        train, val = train_test_split(data[data.hospital != hospital], test_size=0.20, random_state=42)
        fold = pd.concat([train.slides.reset_index(drop=True).rename("train"),
                          val.slides.reset_index(drop=True).rename("val"),
                          test.rename("test")], axis=1).fillna("")
        fold.to_csv(out / f"splits_{idx}.csv")
    print(f"Wrote {len(folds)} folds to {out}")


if __name__ == "__main__":
    main()

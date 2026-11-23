#!/usr/bin/env python3
"""Per-slide cluster composition (from notebook "virchow _kmeans20_clin_var_bubble.ipynb", cells 2-7).

For every slide in the clinical file, reads <labels_dir>/<slide>_Virchow_kmeans_labels.csv
(written by kmeans_inference.py) and stores the fraction of the slide's tiles in each of the
20 Virchow clusters. Output columns: 0..19, slides (one row per slide = patient).
Excluded slides are removed later by the analyses (meta_file/slides_to_remove.csv).
"""

import argparse

import pandas as pd
from tqdm import tqdm

N_CLUSTERS = 20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clinical", required=True, help="clinical CSV with 'Primary_scanned_section'")
    parser.add_argument("--labels_dir", required=True)
    parser.add_argument("--output", required=True, help="e.g. cluster_percentage.csv")
    args = parser.parse_args()

    slides = pd.read_csv(args.clinical)["Primary_scanned_section"].tolist()
    rows = []
    for slide in tqdm(slides):
        labels = pd.read_csv(f"{args.labels_dir}/{slide}_Virchow_kmeans_labels.csv")["cluster"]
        fractions = labels.value_counts(normalize=True).reindex(range(N_CLUSTERS), fill_value=0.0)
        rows.append([*fractions.tolist(), slide])
    out = pd.DataFrame(rows, columns=[str(i) for i in range(N_CLUSTERS)] + ["slides"])
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out):,} slides to {args.output}")


if __name__ == "__main__":
    main()

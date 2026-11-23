#!/usr/bin/env python3
"""Draw the ~1M-tile subsample used for K-means and UMAP (from notebook subsample_1M.ipynb).

Step 1 samples tiles from a reference model's embeddings (UNI in the study): for each slide, a
random number of tiles between 0 and 2x its expected share of 1M (expected share =
n_tiles x 1,000,000 / total tiles in the cohort) is drawn, so the total is ~1M.
Step 2 takes exactly the same tiles (patient, x, y) from every other model's embeddings.
Each output parquet has the D embedding columns followed by patient, x, y.

Note: the original notebook did not set a random seed, so the study subsample cannot be redrawn
exactly; use the saved subsample files to reproduce the paper. --seed makes new runs repeatable.

Embedding CSVs: <embeddings_root>/<model>/<site>/<slide>_*.csv (first column = row index,
then the embedding columns and `patient`). Tile coordinates: <coords_dir>/<slide>_coords.csv.
"""

import argparse
import random
from glob import glob
from pathlib import Path

import pandas as pd
import tqdm

TOTAL_TILES = 16_517_645  # tiles in the full cohort (all slides, all tissue tiles)
TARGET = 1_000_000


def read_embeddings_coords(path, coords_dir):
    embedding_df = pd.read_csv(path).iloc[:, 1:]
    name = Path(path).name.split(".")[0].split("_")[0]
    coord_df = pd.read_csv(Path(coords_dir) / f"{name}_coords.csv")
    return pd.concat([embedding_df, coord_df[["x", "y"]]], axis=1)


def sample_reference(files, coords_dir):
    parts = []
    for path in tqdm.tqdm(files):
        df = read_embeddings_coords(path, coords_dir)
        expected = len(df) * TARGET / TOTAL_TILES
        n = random.randint(0, round(2 * expected))
        if n:
            parts.append(df.sample(n=n))
    return pd.concat(parts, ignore_index=True)


def match_tiles(reference, files, coords_dir):
    keys = ["patient", "x", "y"]
    tiles = reference[keys]
    parts = []
    for patient in tqdm.tqdm(sorted(set(tiles.patient))):
        path = next((f for f in files if patient in f), None)
        if path is None:
            print(f"[WARN] no embedding file for {patient}")
            continue
        emb = read_embeddings_coords(path, coords_dir).set_index(keys)
        parts.append(tiles[tiles.patient.eq(patient)].set_index(keys).join(emb, how="left").reset_index())
    out = pd.concat(parts, ignore_index=True)
    cols = out.columns.tolist()
    return out[cols[3:] + cols[:3]]  # embeddings first, then patient, x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_root", required=True)
    parser.add_argument("--coords_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--reference_model", default="UNI")
    parser.add_argument("--models", nargs="+", default=["Hoptimusv0", "Virchow"])
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ref_files = glob(f"{args.embeddings_root}/{args.reference_model}/*/*csv")
    reference = sample_reference(ref_files, args.coords_dir)
    reference.to_parquet(out / f"{args.reference_model}_subsampled_1M.parquet", index=False)
    print(f"{args.reference_model}: {len(reference):,} tiles")
    for model in args.models:
        files = glob(f"{args.embeddings_root}/{model}/*/*csv")
        matched = match_tiles(reference, files, args.coords_dir)
        matched.to_parquet(out / f"{model}_subsampled_1M.parquet", index=False)
        print(f"{model}: {len(matched):,} tiles")


if __name__ == "__main__":
    main()

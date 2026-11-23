#!/usr/bin/env python3
"""Silhouette score across k = 20-100 for all six foundation models, one protocol.

Uses exactly the protocol of calculate_all_model_silhouettes.py (k = 20 comparison):
five random samples of 10,000 tiles (seeds 42, 0, 12345, 2023, 99) from each model's
1M-tile subsample; labels from the saved kmeans/<model>/kmeans_model_<k>.pkl (no
refitting); exact Euclidean silhouette in the unstandardized embedding space. Sampling
avoids the O(n^2) cost of a silhouette over 1M tiles and puts every model on the same
footing (kmeans.py's original UNI/H-optimus/Virchow values used all 1M tiles; the k = 20
values agree with the sampled ones to within 0.001).
Output: kmeans/silhouette_sweep/silhouette_by_k_by_seed.csv and silhouette_by_k_summary.csv
"""

import gc
import pickle
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn import set_config
from sklearn.metrics import silhouette_samples

from calculate_all_model_silhouettes import (MODELS, ROOT, SAMPLE_SIZE, SEEDS,
                                             load_union_sample, sample_indices)

K_VALUES = [5, 10] + list(range(20, 101, 10))  # k = 5, 10 added 2026-10-30
OUT = ROOT / "silhouette_sweep_k5_100"  # k = 5-100; the k = 20-100 run stays in silhouette_sweep/


def main():
    OUT.mkdir(exist_ok=True)
    set_config(working_memory=512)  # as in calculate_all_model_silhouettes.py
    rows = []
    for display, directory, dimensions in MODELS:
        missing = [k for k in K_VALUES if not (ROOT / directory / f"kmeans_model_{k}.pkl").exists()]
        if missing:
            raise FileNotFoundError(f"{directory}: no K-means model for k = {missing}")
        path = ROOT / f"{directory}_subsampled_1M.parquet"
        n_rows = pq.ParquetFile(path).metadata.num_rows
        columns = [str(i) for i in range(dimensions)]
        indices = sample_indices(n_rows, SAMPLE_SIZE)
        embeddings, positions = load_union_sample(path, columns, indices)
        for k in K_VALUES:
            with open(ROOT / directory / f"kmeans_model_{k}.pkl", "rb") as handle:
                kmeans = pickle.load(handle)
            if kmeans.cluster_centers_.shape != (k, dimensions):
                raise ValueError(f"{directory} k={k}: centers {kmeans.cluster_centers_.shape}")
            for seed in SEEDS:
                matrix = embeddings[positions[seed]].astype(kmeans.cluster_centers_.dtype, copy=False)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="X does not have valid feature names")
                    labels = kmeans.predict(matrix)
                values = silhouette_samples(matrix, labels, metric="euclidean")
                rows.append({"model": display, "model_directory": directory, "k": k,
                             "seed": seed, "sample_size": SAMPLE_SIZE,
                             "silhouette_score": float(values.mean()),
                             "negative_silhouette_fraction": float(np.mean(values < 0)),
                             "clusters_present": int(len(np.unique(labels)))})
            print(f"{display} k={k}: silhouette "
                  f"{np.mean([r['silhouette_score'] for r in rows[-len(SEEDS):]]):.4f}", flush=True)
        del embeddings
        gc.collect()
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / "silhouette_by_k_by_seed.csv", index=False)
    # Built-in check: k = 20 uses the same samples and models as the k = 20 comparison,
    # so it must reproduce those values exactly.
    reference = pd.read_csv(ROOT / "silhouette_comparison_k20" / "silhouette_scores_by_seed.csv")
    check = scores[scores.k == 20].merge(reference, on=["model", "seed"], suffixes=("", "_ref"))
    if len(check) != len(SEEDS) * len(MODELS) or \
            not np.allclose(check.silhouette_score, check.silhouette_score_ref, atol=1e-6):
        raise ValueError("k = 20 silhouettes do not reproduce silhouette_comparison_k20")
    print("k = 20 check passed: identical to silhouette_comparison_k20", flush=True)
    summary = scores.groupby(["model", "k"]).silhouette_score.agg(["mean", "std"]).reset_index()
    summary.to_csv(OUT / "silhouette_by_k_summary.csv", index=False)
    print(summary.pivot(index="k", columns="model", values="mean").round(4).to_string())


if __name__ == "__main__":
    main()

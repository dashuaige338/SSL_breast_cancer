#!/usr/bin/env python3
"""Compare the final k=20 clusterings across pathology foundation models."""

from __future__ import annotations

import argparse
import gc
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn import set_config
from sklearn.metrics import silhouette_samples


ROOT = Path(__file__).resolve().parent / "kmeans"
OUT = ROOT / "silhouette_comparison_k20"
SEEDS = [42, 0, 12345, 2023, 99]
SAMPLE_SIZE = 10_000
MODELS = [
    ("H-optimus-0", "Hoptimusv0", 1536),
    ("UNI", "UNI", 1536),
    ("Virchow", "Virchow", 2560),
    ("CONCH", "CONCH", 768),
    ("Prov-GigaPath", "PROVGIGAPATH", 1536),
    ("KEEP", "KEEP", 768),
]


def sample_indices(n_rows: int, sample_size: int):
    if sample_size >= n_rows:
        raise ValueError(f"Sample size {sample_size:,} must be smaller than {n_rows:,}")
    return {
        seed: np.sort(np.random.default_rng(seed).choice(n_rows, sample_size, replace=False))
        for seed in SEEDS
    }


def load_union_sample(path: Path, feature_columns: list[str], indices_by_seed):
    """Stream one parquet scan and retain only rows selected by any seed."""
    parquet = pq.ParquetFile(path)
    union = np.unique(np.concatenate(list(indices_by_seed.values())))
    sampled = np.empty((len(union), len(feature_columns)), dtype=np.float32)
    offset = 0
    for batch in parquet.iter_batches(batch_size=2_048, columns=feature_columns):
        end = offset + batch.num_rows
        left = np.searchsorted(union, offset, side="left")
        right = np.searchsorted(union, end, side="left")
        if right > left:
            matrix = batch.to_pandas().to_numpy(dtype=np.float32, copy=False)
            sampled[left:right] = matrix[union[left:right] - offset]
        offset = end
    if offset != parquet.metadata.num_rows:
        raise RuntimeError(f"Read {offset:,} rows from {path}; expected {parquet.metadata.num_rows:,}")
    if not np.isfinite(sampled).all():
        raise ValueError(f"Non-finite values found in sampled embeddings from {path}")
    positions = {
        seed: np.searchsorted(union, indices) for seed, indices in indices_by_seed.items()
    }
    return sampled, positions


def calculate_model(display_name, model_name, dimensions, sample_size):
    feature_path = ROOT / f"{model_name}_subsampled_1M.parquet"
    model_path = ROOT / model_name / "kmeans_model_20.pkl"
    parquet = pq.ParquetFile(feature_path)
    n_rows = parquet.metadata.num_rows
    feature_columns = [str(index) for index in range(dimensions)]
    missing = set(feature_columns).difference(parquet.schema.names)
    if missing:
        raise ValueError(f"{model_name} is missing {len(missing)} expected feature columns")

    with model_path.open("rb") as handle:
        kmeans = pickle.load(handle)
    if kmeans.cluster_centers_.shape != (20, dimensions):
        raise ValueError(
            f"Unexpected k-means shape for {model_name}: {kmeans.cluster_centers_.shape}"
        )

    indices = sample_indices(n_rows, sample_size)
    embeddings, positions = load_union_sample(feature_path, feature_columns, indices)
    overall_rows = []
    cluster_rows = []
    for seed in SEEDS:
        matrix = embeddings[positions[seed]].astype(kmeans.cluster_centers_.dtype, copy=False)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            labels = kmeans.predict(matrix)
        counts = np.bincount(labels, minlength=20)
        if np.any(counts == 0):
            absent = np.flatnonzero(counts == 0).tolist()
            raise ValueError(f"{model_name}, seed {seed}: absent clusters {absent}")
        values = silhouette_samples(matrix, labels, metric="euclidean")
        overall_rows.append(
            {
                "model": display_name,
                "model_directory": model_name,
                "seed": seed,
                "sample_size": sample_size,
                "source_population_rows": n_rows,
                "dimensions": dimensions,
                "clusters": 20,
                "metric": "euclidean",
                "silhouette_score": float(values.mean()),
                "silhouette_sd_across_tiles": float(values.std(ddof=1)),
                "negative_silhouette_fraction": float(np.mean(values < 0)),
                "minimum_cluster_sample_n": int(counts.min()),
                "maximum_cluster_sample_n": int(counts.max()),
            }
        )
        for cluster in range(20):
            cluster_values = values[labels == cluster]
            cluster_rows.append(
                {
                    "model": display_name,
                    "model_directory": model_name,
                    "seed": seed,
                    "cluster": cluster,
                    "n": len(cluster_values),
                    "mean_silhouette": float(cluster_values.mean()),
                    "median_silhouette": float(np.median(cluster_values)),
                    "negative_silhouette_fraction": float(np.mean(cluster_values < 0)),
                }
            )
        print(
            f"{display_name}, seed {seed}: silhouette={values.mean():.6f}, "
            f"negative={np.mean(values < 0):.3f}",
            flush=True,
        )
    del embeddings
    gc.collect()
    return pd.DataFrame(overall_rows), pd.DataFrame(cluster_rows)


def summarize(scores):
    rows = []
    for model, frame in scores.groupby("model", sort=False):
        values = frame.silhouette_score.to_numpy()
        mean = values.mean()
        sd = values.std(ddof=1)
        se = sd / np.sqrt(len(values))
        # t(4), 97.5th percentile; five deterministic random samples.
        margin = 2.776445 * se
        rows.append(
            {
                "model": model,
                "n_seeds": len(values),
                "sample_size_per_seed": int(frame.sample_size.iloc[0]),
                "mean_silhouette": mean,
                "sd_across_seeds": sd,
                "ci95_low": mean - margin,
                "ci95_high": mean + margin,
                "minimum_seed_score": values.min(),
                "maximum_seed_score": values.max(),
                "mean_negative_silhouette_fraction": frame.negative_silhouette_fraction.mean(),
            }
        )
    return pd.DataFrame(rows)


def write_readme():
    text = """# Foundation-model silhouette comparison

This analysis evaluates the final saved k-means solutions with k=20 for H-optimus-0, UNI, Virchow, CONCH, Prov-GigaPath, and KEEP.

For each model, five samples of 10,000 tiles were drawn without replacement using seeds 42, 0, 12345, 2023, and 99. The saved `kmeans_model_20.pkl` assigned cluster labels; k-means was not refitted. Exact Euclidean silhouette values were calculated within each sample in the original embedding space. Embeddings were not standardized because the final k-means models were fitted to the unstandardized representations.

The six stored embedding populations are separate representative samples and do not contain an identical set of tiles. Every model uses the same sample size, seeds, number of clusters, metric, and evaluation procedure. The mean and SD across seeds summarize sampling variability; the 95% interval in the summary table is based on the five seed-level estimates.

Higher silhouette scores indicate greater within-cluster cohesion relative to separation from the nearest alternative cluster. Scores near zero indicate overlapping clusters, and negative tile-level values indicate that a tile is closer on average to another cluster than to its assigned cluster.

## Files

- `foundation_model_silhouette_k20.*`: comparison figure in PNG, SVG, and PDF formats.
- `silhouette_scores_by_seed.csv`: overall results for every model and seed.
- `silhouette_summary.csv`: mean, SD, 95% interval, range, and negative-score fraction by model.
- `silhouette_by_cluster.csv`: per-cluster diagnostics for every model and seed.
"""
    (OUT / "README.md").write_text(text)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--force", action="store_true", help="Recompute completed per-model files")
    return parser.parse_args()


def main():
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    set_config(working_memory=512)
    all_scores = []
    all_clusters = []
    for display_name, model_name, dimensions in MODELS:
        score_path = OUT / f"{model_name}_scores.csv"
        cluster_path = OUT / f"{model_name}_cluster_scores.csv"
        if score_path.exists() and cluster_path.exists() and not args.force:
            scores = pd.read_csv(score_path)
            clusters = pd.read_csv(cluster_path)
            expected = set(SEEDS)
            if set(scores.seed) != expected or not (scores.sample_size == args.sample_size).all():
                raise ValueError(f"Existing checkpoint for {model_name} does not match this run")
            print(f"Using completed checkpoint for {display_name}", flush=True)
        else:
            print(f"Calculating {display_name}", flush=True)
            scores, clusters = calculate_model(
                display_name, model_name, dimensions, args.sample_size
            )
            scores.to_csv(score_path, index=False)
            clusters.to_csv(cluster_path, index=False)
        all_scores.append(scores)
        all_clusters.append(clusters)

    scores = pd.concat(all_scores, ignore_index=True)
    clusters = pd.concat(all_clusters, ignore_index=True)
    summary = summarize(scores)
    scores.to_csv(OUT / "silhouette_scores_by_seed.csv", index=False)
    clusters.to_csv(OUT / "silhouette_by_cluster.csv", index=False)
    summary.to_csv(OUT / "silhouette_summary.csv", index=False)
    write_readme()
    print(summary.to_string(index=False), flush=True)
    print(f"Saved results to {OUT}", flush=True)


if __name__ == "__main__":
    main()

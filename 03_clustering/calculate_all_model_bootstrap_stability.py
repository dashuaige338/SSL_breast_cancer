#!/usr/bin/env python3
"""Bootstrap stability of final k=20 foundation-model clusterings."""

from __future__ import annotations

import argparse
import gc
import itertools
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix


ROOT = Path(__file__).resolve().parent / "kmeans"
OUT = ROOT / "bootstrap_stability_k20"
MODELS = [
    ("H-optimus-0", "Hoptimusv0", 1536),
    ("UNI", "UNI", 1536),
    ("Virchow", "Virchow", 2560),
    ("CONCH", "CONCH", 768),
    ("Prov-GigaPath", "PROVGIGAPATH", 1536),
    ("KEEP", "KEEP", 768),
]
BOOTSTRAP_SEEDS = [42, 0, 12345, 2023, 99, 7, 314, 2718, 8675309, 2026]
POOL_SEED = 1729
TRAINING_POOL_SIZE = 20_000
VALIDATION_SIZE = 10_000
N_INIT = 10


def load_selected_rows(path, feature_columns, selections):
    """Read selected parquet rows in a single sequential scan."""
    parquet = pq.ParquetFile(path)
    union = np.unique(np.concatenate(list(selections.values())))
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
        raise RuntimeError(f"Read {offset:,} rows; expected {parquet.metadata.num_rows:,}")
    if not np.isfinite(sampled).all():
        raise ValueError(f"Non-finite embeddings in {path}")
    positions = {key: np.searchsorted(union, value) for key, value in selections.items()}
    return sampled, positions


def match_cluster_jaccard(reference, bootstrap, display_name, model_name, replicate, seed):
    table = contingency_matrix(reference, bootstrap, sparse=False)
    reference_assignment, bootstrap_assignment = linear_sum_assignment(-table)
    bootstrap_for_reference = dict(zip(reference_assignment, bootstrap_assignment))
    rows = []
    for reference_cluster in range(20):
        bootstrap_cluster = bootstrap_for_reference[reference_cluster]
        ref_mask = reference == reference_cluster
        boot_mask = bootstrap == bootstrap_cluster
        intersection = int(np.sum(ref_mask & boot_mask))
        union = int(np.sum(ref_mask | boot_mask))
        rows.append(
            {
                "model": display_name,
                "model_directory": model_name,
                "replicate": replicate,
                "seed": seed,
                "reference_cluster": reference_cluster,
                "matched_bootstrap_cluster": int(bootstrap_cluster),
                "reference_cluster_n": int(ref_mask.sum()),
                "bootstrap_cluster_n": int(boot_mask.sum()),
                "intersection_n": intersection,
                "union_n": union,
                "jaccard": intersection / union if union else np.nan,
                "precision": intersection / int(boot_mask.sum()) if boot_mask.any() else np.nan,
                "recall": intersection / int(ref_mask.sum()) if ref_mask.any() else np.nan,
            }
        )
    return rows


def calculate_model(display_name, model_name, dimensions, pool_size, validation_size, n_init):
    feature_path = ROOT / f"{model_name}_subsampled_1M.parquet"
    model_path = ROOT / model_name / "kmeans_model_20.pkl"
    parquet = pq.ParquetFile(feature_path)
    n_rows = parquet.metadata.num_rows
    if pool_size + validation_size > n_rows:
        raise ValueError(f"Requested samples exceed the {model_name} population")

    rng = np.random.default_rng(POOL_SEED)
    selected = rng.choice(n_rows, pool_size + validation_size, replace=False)
    selections = {
        "validation": np.sort(selected[:validation_size]),
        "training_pool": np.sort(selected[validation_size:]),
    }
    feature_columns = [str(index) for index in range(dimensions)]
    embeddings, positions = load_selected_rows(feature_path, feature_columns, selections)
    validation = embeddings[positions["validation"]]
    training_pool = embeddings[positions["training_pool"]]

    with model_path.open("rb") as handle:
        reference_model = pickle.load(handle)
    if reference_model.cluster_centers_.shape != (20, dimensions):
        raise ValueError(
            f"Unexpected reference model shape for {model_name}: "
            f"{reference_model.cluster_centers_.shape}"
        )
    reference_validation = validation.astype(reference_model.cluster_centers_.dtype, copy=False)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        reference_labels = reference_model.predict(reference_validation)
    if len(np.unique(reference_labels)) != 20:
        raise ValueError(f"Reference labels for {model_name} do not contain all 20 clusters")

    reference_rows = []
    cluster_rows = []
    predictions = {}
    for replicate, seed in enumerate(BOOTSTRAP_SEEDS, start=1):
        bootstrap_rng = np.random.default_rng(seed)
        bootstrap_positions = bootstrap_rng.choice(pool_size, pool_size, replace=True)
        bootstrap_matrix = training_pool[bootstrap_positions]
        model = KMeans(
            n_clusters=20,
            n_init=n_init,
            max_iter=300,
            random_state=seed,
            algorithm="lloyd",
        )
        model.fit(bootstrap_matrix)
        labels = model.predict(validation)
        predictions[replicate] = labels
        ari = adjusted_rand_score(reference_labels, labels)
        unique_training = len(np.unique(bootstrap_positions))
        reference_rows.append(
            {
                "model": display_name,
                "model_directory": model_name,
                "replicate": replicate,
                "seed": seed,
                "k": 20,
                "training_pool_size": pool_size,
                "bootstrap_draws": pool_size,
                "unique_training_tiles": unique_training,
                "validation_size": validation_size,
                "dimensions": dimensions,
                "n_init": n_init,
                "ari_vs_final_solution": ari,
                "bootstrap_inertia_per_draw": float(model.inertia_ / pool_size),
                "bootstrap_iterations": int(model.n_iter_),
            }
        )
        cluster_rows.extend(
            match_cluster_jaccard(
                reference_labels, labels, display_name, model_name, replicate, seed
            )
        )
        print(
            f"{display_name}, bootstrap {replicate}/{len(BOOTSTRAP_SEEDS)}: "
            f"ARI vs final={ari:.5f}, unique training tiles={unique_training:,}",
            flush=True,
        )

    pairwise_rows = []
    for first, second in itertools.combinations(predictions, 2):
        pairwise_rows.append(
            {
                "model": display_name,
                "model_directory": model_name,
                "replicate_a": first,
                "replicate_b": second,
                "seed_a": BOOTSTRAP_SEEDS[first - 1],
                "seed_b": BOOTSTRAP_SEEDS[second - 1],
                "pairwise_ari": adjusted_rand_score(predictions[first], predictions[second]),
            }
        )

    del embeddings, validation, training_pool, predictions
    gc.collect()
    return (
        pd.DataFrame(reference_rows),
        pd.DataFrame(pairwise_rows),
        pd.DataFrame(cluster_rows),
    )


def confidence_interval(values):
    values = np.asarray(values, dtype=float)
    mean = values.mean()
    sd = values.std(ddof=1)
    # t(9), 97.5th percentile for ten bootstrap replicates.
    margin = 2.262157 * sd / np.sqrt(len(values))
    return mean, sd, mean - margin, mean + margin


def summarize(reference, pairwise, clusters):
    rows = []
    for model, frame in reference.groupby("model", sort=False):
        pair = pairwise.loc[pairwise.model == model]
        cluster = clusters.loc[clusters.model == model]
        ref_mean, ref_sd, ref_low, ref_high = confidence_interval(frame.ari_vs_final_solution)
        rows.append(
            {
                "model": model,
                "bootstrap_replicates": len(frame),
                "training_pool_size": int(frame.training_pool_size.iloc[0]),
                "validation_size": int(frame.validation_size.iloc[0]),
                "mean_ari_vs_final": ref_mean,
                "sd_ari_vs_final": ref_sd,
                "ci95_low_ari_vs_final": ref_low,
                "ci95_high_ari_vs_final": ref_high,
                "minimum_ari_vs_final": frame.ari_vs_final_solution.min(),
                "maximum_ari_vs_final": frame.ari_vs_final_solution.max(),
                "mean_pairwise_bootstrap_ari": pair.pairwise_ari.mean(),
                "sd_pairwise_bootstrap_ari": pair.pairwise_ari.std(ddof=1),
                "minimum_pairwise_bootstrap_ari": pair.pairwise_ari.min(),
                "maximum_pairwise_bootstrap_ari": pair.pairwise_ari.max(),
                "mean_cluster_jaccard": cluster.jaccard.mean(),
                "minimum_cluster_mean_jaccard": cluster.groupby("reference_cluster").jaccard.mean().min(),
            }
        )
    return pd.DataFrame(rows)


def write_readme(args):
    text = f"""# Bootstrap stability of foundation-model clusters

This analysis evaluates the final k=20 clusterings for H-optimus-0, UNI, Virchow, CONCH, Prov-GigaPath, and KEEP.

For each model, a fixed development pool of {args.pool_size:,} tiles and a disjoint validation set of {args.validation_size:,} tiles were sampled from the stored embedding population. Ten bootstrap training sets of {args.pool_size:,} draws were generated with replacement from the development pool. A k=20 k-means model was fitted to every bootstrap sample using `n_init={args.n_init}` and up to 300 iterations. Each bootstrap model and the saved final model assigned the same held-out validation tiles.

Adjusted Rand index (ARI) measures label-permutation-invariant agreement. `ARI vs final` compares each bootstrap partition with the saved final solution. `Pairwise bootstrap ARI` compares every pair of the ten bootstrap solutions. Cluster-level Jaccard values were calculated after one-to-one Hungarian matching of bootstrap clusters to final clusters.

ARI equals 1 for identical partitions, has an expected value near 0 for random agreement, and can be negative. Higher ARI and Jaccard values indicate more reproducible clusters.

The stored embedding populations differ slightly across models, but the sample sizes, seeds, k, fitting parameters, and evaluation procedure are identical.

## Files

- `foundation_model_bootstrap_stability_k20.*`: primary comparison figure.
- `bootstrap_stability_summary.csv`: model-level summary.
- `bootstrap_vs_final.csv`: ten ARI values per model against the saved final clustering.
- `bootstrap_pairwise_ari.csv`: all 45 pairwise bootstrap comparisons per model.
- `bootstrap_cluster_jaccard.csv`: cluster-level matched Jaccard, precision, and recall.
"""
    (OUT / "README.md").write_text(text)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-size", type=int, default=TRAINING_POOL_SIZE)
    parser.add_argument("--validation-size", type=int, default=VALIDATION_SIZE)
    parser.add_argument("--n-init", type=int, default=N_INIT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    all_reference = []
    all_pairwise = []
    all_clusters = []
    for display_name, model_name, dimensions in MODELS:
        ref_path = OUT / f"{model_name}_vs_final.csv"
        pair_path = OUT / f"{model_name}_pairwise.csv"
        cluster_path = OUT / f"{model_name}_cluster_jaccard.csv"
        if all(path.exists() for path in (ref_path, pair_path, cluster_path)) and not args.force:
            reference = pd.read_csv(ref_path)
            pairwise = pd.read_csv(pair_path)
            clusters = pd.read_csv(cluster_path)
            if (
                len(reference) != len(BOOTSTRAP_SEEDS)
                or not (reference.training_pool_size == args.pool_size).all()
                or not (reference.validation_size == args.validation_size).all()
                or not (reference.n_init == args.n_init).all()
            ):
                raise ValueError(f"Existing checkpoint for {model_name} does not match this run")
            print(f"Using completed checkpoint for {display_name}", flush=True)
        else:
            print(f"Calculating {display_name}", flush=True)
            reference, pairwise, clusters = calculate_model(
                display_name,
                model_name,
                dimensions,
                args.pool_size,
                args.validation_size,
                args.n_init,
            )
            reference.to_csv(ref_path, index=False)
            pairwise.to_csv(pair_path, index=False)
            clusters.to_csv(cluster_path, index=False)
        all_reference.append(reference)
        all_pairwise.append(pairwise)
        all_clusters.append(clusters)

    reference = pd.concat(all_reference, ignore_index=True)
    pairwise = pd.concat(all_pairwise, ignore_index=True)
    clusters = pd.concat(all_clusters, ignore_index=True)
    summary = summarize(reference, pairwise, clusters)
    reference.to_csv(OUT / "bootstrap_vs_final.csv", index=False)
    pairwise.to_csv(OUT / "bootstrap_pairwise_ari.csv", index=False)
    clusters.to_csv(OUT / "bootstrap_cluster_jaccard.csv", index=False)
    summary.to_csv(OUT / "bootstrap_stability_summary.csv", index=False)
    write_readme(args)
    print(summary.to_string(index=False), flush=True)
    print(f"Saved results to {OUT}", flush=True)


if __name__ == "__main__":
    main()

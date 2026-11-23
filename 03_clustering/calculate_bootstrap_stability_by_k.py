#!/usr/bin/env python3
"""Bootstrap stability for any k, same protocol as calculate_all_model_bootstrap_stability.py.

That script (left unchanged) is fixed to k = 20. This one takes --k and evaluates the saved
kmeans/<model>/kmeans_model_<k>.pkl with the identical design: a 20,000-tile development pool
and a disjoint 10,000-tile validation set (POOL_SEED), ten bootstrap K-means refits
(BOOTSTRAP_SEEDS, n_init = 10, lloyd, max_iter 300), ARI of each refit against the saved
model on the validation tiles, pairwise ARI among refits, and Hungarian-matched per-cluster
Jaccard.
Output (new folder): kmeans/bootstrap_stability_k<k>/  (--k 20 writes
kmeans/bootstrap_stability_k20_rerun/ and is checked against bootstrap_stability_k20/).
"""

import argparse
import gc
import itertools
import pickle
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix

from calculate_all_model_bootstrap_stability import (BOOTSTRAP_SEEDS, MODELS, N_INIT, POOL_SEED, ROOT,
                                                     TRAINING_POOL_SIZE, VALIDATION_SIZE,
                                                     load_selected_rows, summarize)


def cluster_jaccard(reference, bootstrap, k, base):
    table = contingency_matrix(reference, bootstrap, sparse=False)
    rows_idx, cols_idx = linear_sum_assignment(-table)
    matched = dict(zip(rows_idx, cols_idx))
    rows = []
    for c in range(k):
        ref_mask, boot_mask = reference == c, bootstrap == matched[c]
        inter, union = int(np.sum(ref_mask & boot_mask)), int(np.sum(ref_mask | boot_mask))
        rows.append({**base, "reference_cluster": c, "matched_bootstrap_cluster": int(matched[c]),
                     "reference_cluster_n": int(ref_mask.sum()), "bootstrap_cluster_n": int(boot_mask.sum()),
                     "intersection_n": inter, "union_n": union, "jaccard": inter / union if union else np.nan})
    return rows


def calculate_model(display, directory, dimensions, k):
    path = ROOT / f"{directory}_subsampled_1M.parquet"
    n_rows = pq.ParquetFile(path).metadata.num_rows
    selected = np.random.default_rng(POOL_SEED).choice(n_rows, TRAINING_POOL_SIZE + VALIDATION_SIZE, replace=False)
    selections = {"validation": np.sort(selected[:VALIDATION_SIZE]),
                  "training_pool": np.sort(selected[VALIDATION_SIZE:])}
    embeddings, positions = load_selected_rows(path, [str(i) for i in range(dimensions)], selections)
    validation, pool = embeddings[positions["validation"]], embeddings[positions["training_pool"]]
    with (ROOT / directory / f"kmeans_model_{k}.pkl").open("rb") as handle:
        final = pickle.load(handle)
    if final.cluster_centers_.shape != (k, dimensions):
        raise ValueError(f"{directory} k={k}: centers {final.cluster_centers_.shape}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        reference = final.predict(validation.astype(final.cluster_centers_.dtype, copy=False))
    if len(np.unique(reference)) != k:
        raise ValueError(f"{directory} k={k}: validation tiles do not cover all clusters")
    ref_rows, cluster_rows, predictions = [], [], {}
    for replicate, seed in enumerate(BOOTSTRAP_SEEDS, start=1):
        draws = np.random.default_rng(seed).choice(TRAINING_POOL_SIZE, TRAINING_POOL_SIZE, replace=True)
        model = KMeans(n_clusters=k, n_init=N_INIT, max_iter=300, random_state=seed, algorithm="lloyd")
        model.fit(pool[draws])
        labels = model.predict(validation)
        predictions[replicate] = labels
        base = {"model": display, "model_directory": directory, "replicate": replicate, "seed": seed}
        ref_rows.append({**base, "k": k, "training_pool_size": TRAINING_POOL_SIZE,
                         "validation_size": VALIDATION_SIZE, "n_init": N_INIT,
                         "ari_vs_final_solution": adjusted_rand_score(reference, labels)})
        cluster_rows.extend(cluster_jaccard(reference, labels, k, base))
        print(f"{display} k={k} bootstrap {replicate}: ARI {ref_rows[-1]['ari_vs_final_solution']:.4f}", flush=True)
    pair_rows = [{"model": display, "model_directory": directory, "replicate_a": a, "replicate_b": b,
                  "pairwise_ari": adjusted_rand_score(predictions[a], predictions[b])}
                 for a, b in itertools.combinations(predictions, 2)]
    del embeddings, validation, pool
    gc.collect()
    return pd.DataFrame(ref_rows), pd.DataFrame(pair_rows), pd.DataFrame(cluster_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, required=True)
    args = parser.parse_args()
    out = ROOT / (f"bootstrap_stability_k{args.k}" + ("_rerun" if args.k == 20 else ""))
    out.mkdir(exist_ok=True)
    results = [calculate_model(d, m, dim, args.k) for d, m, dim in MODELS]
    reference, pairwise, clusters = (pd.concat(parts, ignore_index=True) for parts in zip(*results))
    reference.to_csv(out / "bootstrap_vs_final.csv", index=False)
    pairwise.to_csv(out / "bootstrap_pairwise_ari.csv", index=False)
    clusters.to_csv(out / "bootstrap_cluster_jaccard.csv", index=False)
    summary = summarize(reference, pairwise, clusters)
    summary.to_csv(out / "bootstrap_stability_summary.csv", index=False)
    print(summary[["model", "mean_ari_vs_final", "mean_pairwise_bootstrap_ari", "mean_cluster_jaccard"]]
          .round(4).to_string(index=False))
    if args.k == 20:
        old = pd.read_csv(ROOT / "bootstrap_stability_k20" / "bootstrap_vs_final.csv")
        check = reference.merge(old, on=["model", "replicate"], suffixes=("", "_old"))
        # Thread count can change K-means in the last digits; a real protocol difference would not stay this small.
        diff = (check.ari_vs_final_solution - check.ari_vs_final_solution_old).abs().max()
        if len(check) != len(old) or diff > 0.01:
            raise ValueError(f"k = 20 rerun does not reproduce bootstrap_stability_k20 (max ARI diff {diff})")
        print(f"k = 20 check passed vs bootstrap_stability_k20 (max ARI diff {diff:.2e})", flush=True)


if __name__ == "__main__":
    main()

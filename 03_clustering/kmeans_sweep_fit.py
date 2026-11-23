#!/usr/bin/env python3
"""Fit one K-means model (one foundation model, one k) on its 1M-tile subsample.

Completes the k = 20-100 sweep of kmeans.py for models that only have k = 20
(CONCH, Prov-GigaPath, KEEP), with the same settings as kmeans.py and
run_canonical_kmeans_umap.py: KMeans(n_clusters=k, n_init=20, random_state=42) on the
unstandardized embeddings in kmeans/<model>_subsampled_1M.parquet.
Saves kmeans/<model>/kmeans_model_<k>.pkl. Never overwrites an existing model (the k = 20
models feed every downstream figure); exits early if the file is already there.
Also writes kmeans/<model>/sweep_fit_k<k>.json (inertia, iterations, Calinski-Harabasz
and Davies-Bouldin on all 1M tiles, fit time).
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

ROOT = Path(__file__).resolve().parent / "kmeans"
DIMENSIONS = {"Hoptimusv0": 1536, "UNI": 1536, "Virchow": 2560, "CONCH": 768,
              "PROVGIGAPATH": 1536, "KEEP": 768}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(DIMENSIONS))
    parser.add_argument("--k", type=int, required=True)
    args = parser.parse_args()
    out = ROOT / args.model
    model_path = out / f"kmeans_model_{args.k}.pkl"
    if model_path.exists():
        print(f"{model_path} already exists; not refitting.", flush=True)
        return

    columns = [str(i) for i in range(DIMENSIONS[args.model])]
    matrix = pq.read_table(ROOT / f"{args.model}_subsampled_1M.parquet", columns=columns) \
        .to_pandas().to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite embedding values")
    print(f"{args.model}: matrix {matrix.shape}; fitting k = {args.k}", flush=True)

    start = time.time()
    kmeans = KMeans(n_clusters=args.k, n_init=20, random_state=42, verbose=0)
    labels = kmeans.fit_predict(matrix)
    fit_seconds = time.time() - start

    temporary = model_path.with_suffix(".pkl.tmp")
    with open(temporary, "wb") as handle:
        pickle.dump(kmeans, handle)
    temporary.replace(model_path)

    info = {"model": args.model, "k": args.k, "n_tiles": int(matrix.shape[0]),
            "inertia": float(kmeans.inertia_), "n_iter": int(kmeans.n_iter_),
            "calinski_harabasz": float(calinski_harabasz_score(matrix, labels)),
            "davies_bouldin": float(davies_bouldin_score(matrix, labels)),
            "min_cluster_size": int(np.bincount(labels).min()),
            "fit_seconds": round(fit_seconds, 1)}
    (out / f"sweep_fit_k{args.k}.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Calinski-Harabasz and Davies-Bouldin at k = 20 for CONCH, Prov-GigaPath and KEEP.

Completes the k = 20-100 CH/DB table: these three models' k = 20 fits came from
run_canonical_kmeans_umap.py, which did not record CH/DB. Uses the saved labels
(kmeans/<model>/canonical_kmeans20_labels.parquet) on all 1M tiles, as kmeans.py and
kmeans_sweep_fit.py did. Writes kmeans/<model>/sweep_fit_k20.json (no refitting).
"""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

ROOT = Path(__file__).resolve().parent / "kmeans"
for model, dims in [("CONCH", 768), ("PROVGIGAPATH", 1536), ("KEEP", 768)]:
    out = ROOT / model / "sweep_fit_k20.json"
    if out.exists():
        print(f"{out} exists; skipping"); continue
    X = pq.read_table(ROOT / f"{model}_subsampled_1M.parquet", columns=[str(i) for i in range(dims)]) \
        .to_pandas().to_numpy(dtype=np.float32, copy=False)
    labels = pq.read_table(ROOT / model / "canonical_kmeans20_labels.parquet", columns=["cluster"])["cluster"].to_numpy()
    if len(labels) != len(X) or len(np.unique(labels)) != 20:
        raise ValueError(f"{model}: {len(labels)} labels for {len(X)} tiles")
    info = {"model": model, "k": 20, "n_tiles": int(len(X)),
            "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
            "davies_bouldin": float(davies_bouldin_score(X, labels)),
            "source": "canonical_kmeans20_labels.parquet (no refit)"}
    out.write_text(json.dumps(info, indent=2)); print(json.dumps(info), flush=True)

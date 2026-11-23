#!/usr/bin/env python3
"""K=20 and UMAP on the exact canonical 969,549-tile sample."""

import argparse
import gc
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import umap
from sklearn.cluster import KMeans


HOME = Path("/gpfs/data/tsirigoslab/home/jz5490")
EMBED_ROOT = Path("/gpfs/data/abl/deepomics/meskerlab/breast_model_embeddings")
KMEANS_ROOT = HOME / "LUMC_breast_cancer/SSL/kmeans"
CANONICAL = KMEANS_ROOT / "Hoptimusv0_subsampled_1M.parquet"

MODEL_CONFIG = {
    "CONCH": (EMBED_ROOT / "TITAN", "_TITAN_CONCH_img_features.csv", 768),
    "PROVGIGAPATH": (
        EMBED_ROOT / "PROVGIGAPATH", "_PROVGIGAPATH_img_features.csv", 1536
    ),
    "KEEP": (EMBED_ROOT / "KEEP", "_KEEP_img_features.csv", 768),
}


def locate_files(root, suffix):
    files = {}
    for path in root.rglob(f"*{suffix}"):
        slide_id = path.name[: -len(suffix)]
        if slide_id in files:
            raise RuntimeError(f"Duplicate embedding files for {slide_id}")
        files[slide_id] = path
    return files


def build_canonical_parquet(model, output, dimensions):
    keys = pq.read_table(CANONICAL, columns=["patient", "x", "y"]).to_pandas()
    if len(keys) != 969_549 or keys.duplicated().any():
        raise RuntimeError("Canonical tile-key file has unexpected size or duplicates")

    root, suffix, _ = MODEL_CONFIG[model]
    files = locate_files(root, suffix)
    missing = sorted(set(keys.patient) - set(files))
    if missing:
        raise RuntimeError(f"Missing {model} slides: {missing}")

    matrix = np.empty((len(keys), dimensions), dtype=np.float32)
    grouped = keys.groupby("patient", sort=False).groups
    for number, (slide_id, positions) in enumerate(grouped.items(), start=1):
        positions = np.asarray(positions, dtype=np.int64)
        coords = pd.read_csv(
            EMBED_ROOT / "coords" / f"{slide_id}_coords.csv", usecols=["x", "y"]
        )
        if coords.duplicated().any():
            raise RuntimeError(f"Duplicate coordinates in {slide_id}")
        index_by_coord = pd.Series(
            np.arange(len(coords), dtype=np.int64),
            index=pd.MultiIndex.from_frame(coords[["x", "y"]]),
        )
        requested = pd.MultiIndex.from_frame(keys.loc[positions, ["x", "y"]])
        row_indices = index_by_coord.reindex(requested)
        if row_indices.isna().any():
            raise RuntimeError(f"Canonical coordinates absent from {slide_id}")

        features = pd.read_csv(
            files[slide_id], usecols=range(1, dimensions + 1), dtype=np.float32
        ).to_numpy(copy=False)
        if len(features) != len(coords):
            raise RuntimeError(
                f"Row mismatch for {slide_id}: embeddings={len(features)}, coords={len(coords)}"
            )
        matrix[positions] = features[row_indices.to_numpy(dtype=np.int64)]
        if number % 100 == 0:
            print(f"{model}: loaded {number}/{len(grouped)} slides", flush=True)

    if not np.isfinite(matrix).all():
        raise RuntimeError(f"Non-finite values found in {model} matrix")

    columns = [str(i) for i in range(dimensions)]
    feature_table = pa.Table.from_arrays(
        [pa.array(matrix[:, i]) for i in range(dimensions)], names=columns
    )
    key_table = pa.Table.from_pandas(keys, preserve_index=False)
    table = pa.Table.from_arrays(
        list(feature_table.columns) + list(key_table.columns),
        names=feature_table.column_names + key_table.column_names,
    )
    temporary = output.with_suffix(".parquet.tmp")
    pq.write_table(table, temporary, compression="snappy", row_group_size=50_000)
    temporary.replace(output)
    print(f"{model}: saved canonical matrix {matrix.shape} to {output}", flush=True)
    return matrix, keys


def load_matrix(path, dimensions):
    columns = [str(i) for i in range(dimensions)]
    table = pq.read_table(path, columns=columns)
    return table.to_pandas().to_numpy(dtype=np.float32, copy=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODEL_CONFIG))
    args = parser.parse_args()
    model = args.model
    dimensions = MODEL_CONFIG[model][2]
    outdir = KMEANS_ROOT / model
    outdir.mkdir(parents=True, exist_ok=True)
    sample_path = KMEANS_ROOT / f"{model}_subsampled_1M.parquet"

    if sample_path.exists():
        matrix = load_matrix(sample_path, dimensions)
        keys = pq.read_table(sample_path, columns=["patient", "x", "y"]).to_pandas()
        print(f"{model}: resumed canonical matrix {matrix.shape}", flush=True)
    else:
        matrix, keys = build_canonical_parquet(model, sample_path, dimensions)

    model_path = outdir / "kmeans_model_20.pkl"
    labels_path = outdir / "canonical_kmeans20_labels.parquet"
    if model_path.exists() and labels_path.exists():
        with open(model_path, "rb") as handle:
            kmeans = pickle.load(handle)
        labels = pq.read_table(labels_path, columns=["cluster"])["cluster"].to_numpy()
    else:
        kmeans = KMeans(n_clusters=20, n_init=20, random_state=42, verbose=1)
        labels = kmeans.fit_predict(matrix)
        temporary_model = model_path.with_suffix(".pkl.tmp")
        with open(temporary_model, "wb") as handle:
            pickle.dump(kmeans, handle)
        temporary_model.replace(model_path)
        label_frame = keys.copy()
        label_frame.insert(0, "cluster", labels.astype(np.int16))
        label_frame.to_parquet(labels_path, index=False)
        print(f"{model}: completed K=20", flush=True)

    umap_path = outdir / "umap_40.parquet"
    if umap_path.exists():
        coordinates = pd.read_parquet(umap_path)[["umap1", "umap2"]].to_numpy()
    else:
        reducer = umap.UMAP(
            n_components=2, n_neighbors=40, min_dist=0.1,
            metric="euclidean", random_state=42, verbose=True,
        )
        coordinates = reducer.fit_transform(matrix)
        umap_frame = keys.copy()
        umap_frame.insert(0, "cluster", labels.astype(np.int16))
        umap_frame.insert(0, "umap2", coordinates[:, 1])
        umap_frame.insert(0, "umap1", coordinates[:, 0])
        umap_frame.to_parquet(umap_path, index=False)
        with open(outdir / "umap_40_model.pkl", "wb") as handle:
            pickle.dump(reducer, handle)
        print(f"{model}: completed UMAP", flush=True)

    del matrix
    gc.collect()
    print(f"{model}: all outputs complete", flush=True)


if __name__ == "__main__":
    main()

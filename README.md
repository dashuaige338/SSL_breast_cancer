# Foundation-model tissue phenotypes and SurvCLAM prognosis in older breast cancer patients

Analysis code for the study of 3,389 patients aged ≥ 70 years from 36 institutions: tile-level
foundation-model embeddings, K-means tissue phenotypes, and attention-based survival models
(SurvCLAM) for 10-year distant recurrence (DR) and overall survival.
Figure-drawing code is not included.

No patient data, clinical files, slide lists, institution names or trained weights are part of
this repository.

## Pipeline

| Step | Folder | What it does |
|---|---|---|
| 1 | `01_tiling/` | Otsu tissue mask (TIAToolbox, 1× thumbnail) and 448 × 448 px tiles at 20× (level 0, 0.39 µm/px), 32 px overlap, ≥ 75% tissue (`get_coords.py`) |
| 2 | `02_feature_extraction/` | Tile embeddings from H-optimus-0, UNI, Virchow, Prov-GigaPath, KEEP and CONCH (`main.py`, `PathologyFoundationModels/`) |
| 3 | `03_clustering/` | ~1M-tile subsample, K-means (k = 5–100), silhouette / Calinski–Harabasz / Davies–Bouldin, bootstrap stability, k = 20 cluster labels per slide, per-patient cluster composition |
| 4 | `04_survclam/` | Survival dataset CSVs, leave-one-institution-out splits (28 folds), SurvCLAM training and evaluation, uniform-attention ablation and tile-level risk scores |

### 1. Tiling
```bash
python 01_tiling/get_coords.py --input_path <mrxs_root> --output_path <coords_dir>
```

### 2. Tile embeddings
```bash
export HF_TOKEN=<your Hugging Face token>   # gated models (H-optimus-0, UNI, Virchow, Prov-GigaPath)
python 02_feature_extraction/main.py --input_path <mrxs_root> --output_path <embeddings_root>/<model> --model Virchow
```
`--model`: `Hoptimusv0`, `UNI`, `Virchow`, `PROVGIGAPATH`, `KEEP`. CONCH tile embeddings were
extracted with `PathologyFoundationModels/run_conch_only.py` on the same tiles. Prov-GigaPath
needs the `gigapath` package (https://github.com/prov-gigapath/prov-gigapath).

### 3. Clustering
```bash
python 03_clustering/subsample_1M.py --embeddings_root <embeddings_root> --coords_dir <coords_dir> --output_dir <kmeans_dir>
python 03_clustering/remove_excluded_slides.py --parquet <kmeans_dir>/Virchow_subsampled_1M.parquet --remove_list slides_to_remove.csv --output <...>
python 03_clustering/kmeans.py ...                       # k = 20-100 fits and metrics (see sbatch files)
python 03_clustering/kmeans_sweep_fit.py ...             # additional k (5, 10, ...)
python 03_clustering/run_canonical_kmeans_umap.py ...    # k = 20 fit and UMAP on the canonical sample
python 03_clustering/calculate_all_model_silhouettes.py
python 03_clustering/calculate_all_model_bootstrap_stability.py
python 03_clustering/calculate_bootstrap_stability_by_k.py --k 5
python 03_clustering/kmeans_inference.py --input_path <embeddings_root> --output_path <labels_dir> --model Virchow --dimensions 2560
python 03_clustering/cluster_percentage.py --clinical <clinical.csv> --labels_dir <labels_dir> --output cluster_percentage.csv
```
The `.sbatch` files are the SLURM jobs used for the sweeps.

### 4. SurvCLAM
```bash
cd 04_survclam
python make_survival_csv.py --slides <slide_list.csv> --clinical <clinical.csv> --output dataset_csv/poet_breast_recurrence_virchow_20x.csv
python remove_excluded_slides_csv.py --input_dir dataset_csv --remove_list slides_to_remove.csv --output_dir dataset_csv_clean
python make_institution_splits.py --clinical <clinical.csv> --hospitals <slides_hospitals.csv> \
       --exclude_file <excluded_institutions.txt> --output_dir splits/poet_breast_recurrence_hospitals_20x
python main.py --task poet_breast_recurrence_virchow_20x --exp_code poet_breast_recurrence_virchow_hospitals_20x \
       --data_root_dir <features_root> --split_dir splits/poet_breast_recurrence_hospitals_20x --k 28 \
       --model_type clam_sb_surv --model_size small --drop_out 0.25 --lr 1e-4 --reg 1e-5 --max_epochs 200 \
       --weighted_sample --bag_loss ce --inst_loss svm --B 8 --bag_weight 0.7 --embed_dim 2560 --seed 1
python eval.py --task poet_breast_recurrence_virchow_20x --models_exp_code poet_breast_recurrence_virchow_hospitals_20x_s1 \
       --save_exp_code poet_breast_recurrence_virchow_hospitals_20x_s1_cv_test --splits_dir splits/poet_breast_recurrence_hospitals_20x \
       --k 28 --split test --embed_dim 2560 --data_root_dir <features_root>
python attention_ablation_dr.py --model virchow    # uniform-attention ablation, tile risk scores
```
`--embed_dim`: 1536 (H-optimus-0, UNI), 2560 (Virchow). SurvCLAM is CLAM_SB with a linear Cox risk
head (`models/model_clam_survival.py`, loss in `utils/core_utils_survival.py`); the slide risk equals
Σ_i w_i s_i with attention weights w_i = softmax(a_i) and tile risk scores s_i.

## Notes
- **Paths.** Several scripts contain the absolute paths of the original compute cluster
  (`/gpfs/...`); change them to your locations.
- **Subsample.** The original 1M-tile subsample was drawn without a fixed random seed;
  `subsample_1M.py --seed` makes new draws repeatable but cannot recreate the original sample.
- **Excluded slides.** Slides of the institution that withdrew consent are removed with
  `remove_excluded_slides*.py` (list of slide IDs not included).
- **Credentials.** Hugging Face access is read from the `HF_TOKEN` environment variable.
- **Licence.** `04_survclam/` is adapted from CLAM (Lu et al., Nat. Biomed. Eng. 2021;
  https://github.com/mahmoodlab/CLAM) and keeps its licence (`04_survclam/LICENSE.md`).

## Environment
Python 3.9; package versions used are in `requirements.txt`.

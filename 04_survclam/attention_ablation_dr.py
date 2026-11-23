#!/usr/bin/env python3
"""Uniform-attention ablation of the 28-fold DR SurvCLAM models (H-optimus-0, Virchow).

For every held-out slide of each institution fold, the trained model is applied twice:
  learned  - normal forward pass (must reproduce the saved fold_<k>.csv risk; checked);
  uniform  - attention replaced by equal weights, i.e. the slide embedding is the plain
             mean of the tile features after the model's feature/attention layers, then
             the same risk head.
The C-index of each version (pooled raw risk and within-fold percentile, as in Fig. 7)
shows how much of the prediction depends on where the model attends.

Tile-level analysis (same pass). Each tile's own risk score s_i = risk_layer(h_i) is signed
(> slide average pushes risk up); the slide risk is sum_i a_i s_i, so a_i s_i is the tile's
contribution. Per slide: risk from the top-20% attention tiles only and from the remaining
80% only (weights renormalised); within-slide Spearman of attention with s_i (risk-like
attention) and with |s_i - mean s| (informative-tile attention). Per slide x Virchow k=20
cluster: tile count, mean attention weight x N and mean s_i (labels from
breast_model_embeddings/kmeans_20/Virchow, same tile order as the features).
Inputs: results/poet_breast_recurrence_{model}_hospitals_20x_s1/s_<k>_checkpoint.pt,
eval_results/EVAL_poet_breast_recurrence_{model}_hospitals_20x_s1_cv_test/fold_<k>.csv,
/gpfs/scratch/jz5490/poet_breast_recurrence_{model}_features_20x/pt_files/<slide>.pt
Output (new folder): eval_results/attention_ablation_dr/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lifelines.utils import concordance_index

from models.model_clam_survival import CLAM_SB_Surv

HERE = Path(__file__).resolve().parent
OUT = HERE / "eval_results" / "attention_ablation_dr"
MODELS = {"hoptimus": 1536, "virchow": 2560}
FEATURES = Path("/gpfs/scratch/jz5490")
LABELS = Path("/gpfs/data/abl/deepomics/meskerlab/breast_model_embeddings/kmeans_20/Virchow")


def load_model(model, fold, device):
    net = CLAM_SB_Surv(embed_dim=MODELS[model], dropout=0.25, size_arg="small", n_classes=1)
    ckpt = torch.load(HERE / f"results/poet_breast_recurrence_{model}_hospitals_20x_s1/s_{fold}_checkpoint.pt",
                      map_location="cpu")
    net.load_state_dict({k.replace(".module", ""): v for k, v in ckpt.items()
                         if "instance_loss_fn" not in k}, strict=True)
    return net.to(device).eval()


@torch.no_grad()
def tile_outputs(net, x):
    """Attention weights a (sum 1) and signed per-tile risk scores s, as numpy arrays."""
    h = net.feature_extractor(x)
    if net.use_gated:
        a, h = net.attention_net(h)
    else:
        a = net.attention_net(h)
    weights = torch.softmax(a.view(-1), dim=0)
    scores = net.risk_layer(h).view(-1)  # linear head: slide risk = sum(weights * scores)
    return weights.double().cpu().numpy(), scores.double().cpu().numpy()


@torch.no_grad()
def risks(net, x):
    w, s = tile_outputs(net, x)
    return float((w * s).sum()), float(s.mean())


def subset_risk(w, s, mask):
    return float((w[mask] * s[mask]).sum() / w[mask].sum()) if mask.any() else np.nan


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation) if len(a) > 2 else np.nan


def cluster_labels(slide, n):
    path = LABELS / f"{slide}_Virchow_kmeans_labels.csv"
    if not path.exists():
        return None
    labels = pd.read_csv(path, usecols=["cluster"])["cluster"].to_numpy()
    return labels if len(labels) == n else None


def cindex(d, col):
    return concordance_index(d.time, -d[col], d.event)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS), required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_dir = HERE / f"eval_results/EVAL_poet_breast_recurrence_{args.model}_hospitals_20x_s1_cv_test"
    feats = FEATURES / f"poet_breast_recurrence_{args.model}_features_20x" / "pt_files"
    rows, cluster_rows = [], []
    for fold in range(28):
        saved = pd.read_csv(eval_dir / f"fold_{fold}.csv")
        net = load_model(args.model, fold, device)
        for r in saved.itertuples():
            x = torch.load(feats / f"{r.slide_id}.pt", map_location="cpu").float().to(device)
            w, sc = tile_outputs(net, x)
            n = len(w)
            top = w >= np.quantile(w, 0.8)
            rows.append({"fold": fold, "slide_id": r.slide_id, "time": r.time, "event": r.event,
                         "saved_risk": r.risk_score, "learned_risk": float((w * sc).sum()),
                         "uniform_risk": float(sc.mean()), "n_tiles": n,
                         "top20_only_risk": subset_risk(w, sc, top),
                         "rest80_only_risk": subset_risk(w, sc, ~top),
                         "rho_attn_vs_score": spearman(w, sc),
                         "rho_attn_vs_abs_dev": spearman(w, np.abs(sc - sc.mean())),
                         "tile_score_sd": float(sc.std())})
            labels = cluster_labels(r.slide_id, n)
            if labels is not None:
                frame = pd.DataFrame({"cluster": labels, "w_rel": w * n, "score": sc})
                for c, g in frame.groupby("cluster"):
                    cluster_rows.append({"slide_id": r.slide_id, "fold": fold, "cluster": int(c),
                                         "n": len(g), "mean_w_rel": g.w_rel.mean(),
                                         "mean_score": g.score.mean()})
        print(f"{args.model} fold {fold}: {len(saved)} slides", flush=True)
    d = pd.DataFrame(rows)
    diff = (d.learned_risk - d.saved_risk).abs().max()
    if diff > 1e-3:
        raise ValueError(f"Learned-attention risk does not reproduce saved predictions (max diff {diff})")
    pd.DataFrame(cluster_rows).to_csv(OUT / f"{args.model}_tile_cluster_contributions.csv", index=False)
    for col in ("learned_risk", "uniform_risk", "top20_only_risk", "rest80_only_risk"):
        d[col.replace("risk", "pct")] = d.groupby("fold")[col].rank(pct=True)
    d.to_csv(OUT / f"{args.model}_ablation_predictions.csv", index=False)
    summary = pd.DataFrame([{
        "model": args.model, "patients": len(d), "events": int(d.event.sum()),
        "max_abs_diff_vs_saved": diff,
        "pooled_c_learned": cindex(d, "learned_risk"), "pooled_c_uniform": cindex(d, "uniform_risk"),
        "within_fold_pct_c_learned": cindex(d, "learned_pct"), "within_fold_pct_c_uniform": cindex(d, "uniform_pct"),
        "within_fold_pct_c_top20_only": cindex(d, "top20_only_pct"),
        "within_fold_pct_c_rest80_only": cindex(d, "rest80_only_pct"),
        "median_rho_attn_vs_score": d.rho_attn_vs_score.median(),
        "median_rho_attn_vs_abs_dev": d.rho_attn_vs_abs_dev.median(),
        "slides_with_cluster_labels": int(pd.DataFrame(cluster_rows).slide_id.nunique()) if cluster_rows else 0,
        "spearman_learned_vs_uniform": d.learned_risk.corr(d.uniform_risk, method="spearman"),
    }])
    summary.to_csv(OUT / f"{args.model}_ablation_summary.csv", index=False)
    print(summary.round(4).T.to_string())


if __name__ == "__main__":
    main()

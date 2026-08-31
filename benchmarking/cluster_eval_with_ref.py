#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def compute_ARI(pred, gt):
    return adjusted_rand_score(gt, pred)


def compute_NMI(pred, gt):
    return normalized_mutual_info_score(gt, pred)


def filter_matched_labels(ref_adata, pred_adata, gt_key, pred_key):
    """
    Filter and align labels from reference and prediction data.

    If ref_adata and pred_adata are the same object, only filters NaN labels.
    Otherwise, first filters spots based on xy coordinates, then removes NaN labels.
    Returns aligned prediction and ground truth labels with matching indices.

    Parameters:
    -----------
    ref_adata : AnnData
        Reference data with ground truth labels
    pred_adata : AnnData
        Prediction data with predicted labels
    gt_key : str
        Key for ground truth labels in ref_adata.obs
    pred_key : str
        Key for predicted labels in pred_adata.obs

    Returns:
    --------
    pred_labels_valid : pd.Series
        Valid predicted labels
    gt_labels_valid : pd.Series
        Valid ground truth labels (aligned with pred_labels_valid)
    """
    # Check if ref_adata and pred_adata are the same object
    if ref_adata is pred_adata:
        # Same adata, no need for xy matching, just filter NaN
        pred_labels = pred_adata.obs[pred_key]
        gt_labels = ref_adata.obs[gt_key]
    else:
        # Different adata, need to match by xy coordinates first
        if "x" not in ref_adata.obs or "y" not in ref_adata.obs:
            ref_adata.obs[["x", "y"]] = ref_adata.obsm["spatial"]
        if "x" not in pred_adata.obs or "y" not in pred_adata.obs:
            pred_adata.obs[["x", "y"]] = pred_adata.obsm["spatial"]
        ref_xy = pd.concat([ref_adata.obs["x"], ref_adata.obs["y"]], axis=1).reset_index()
        pred_xy = pd.concat([pred_adata.obs["x"], pred_adata.obs["y"]], axis=1).reset_index()
        valid_xy = pd.merge(ref_xy, pred_xy, how="outer", on=["x", "y"], indicator=True)
        ref_xy_valid_index = valid_xy[valid_xy["_merge"] == "both"]["index_x"]
        pred_xy_valid_index = valid_xy[valid_xy["_merge"] == "both"]["index_y"]

        # Get labels for xy-matched spots
        pred_labels = pred_adata.obs.loc[pred_xy_valid_index, pred_key]
        gt_labels = ref_adata.obs.loc[ref_xy_valid_index, gt_key]

    # Filter out NaN labels and keep indices aligned
    label_valid_mask = ~(pred_labels.isna() | gt_labels.isna())
    pred_labels_valid = pred_labels[label_valid_mask]
    gt_labels_valid = gt_labels[label_valid_mask]

    return pred_labels_valid, gt_labels_valid


########################### main ######################################


def main(anno_path, pred_path, outpath, pred_keys, gt_key):
    adata = sc.read_h5ad(anno_path)
    pred_adata = sc.read_h5ad(pred_path)

    output_dfs = []
    for pred_key in pred_keys:
        print(f"calculating metrics for {pred_key}...")
        # Filter and align labels
        pred_labels_valid, gt_labels_valid = filter_matched_labels(adata, pred_adata, gt_key, pred_key)

        ## Accuracy
        ari = compute_ARI(pred_labels_valid, gt_labels_valid)
        print("ARI: ", ari)
        nmi = compute_NMI(pred_labels_valid, gt_labels_valid)
        print("NMI: ", nmi)
        output_df = pd.DataFrame()
        output_df["type"] = ["Accuracy", "Accuracy"]
        output_df["name"] = ["ARI(up)", "NMI(up)"]
        output_df["value"] = [ari, nmi]
        output_df.insert(loc=0, column="pred_key", value=pred_key)
        output_dfs.append(output_df)
    pd.concat(output_dfs).to_csv(outpath, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input", type=str, required=True, help="Cluster file with .h5ad format")
    parser.add_argument(
        "-r", "--reference", type=str, default=None, help="Ground Truth Annotation file with .h5ad format"
    )
    parser.add_argument("--gt_key", type=str, required=True, help="The ground truth key in reference")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output tsv file")
    parser.add_argument(
        "--pred_keys",
        type=str,
        required=True,
        nargs="+",
        help="Predict Cluster key in stored in anndata.obs. Multiple keys can be separated by space",
    )

    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    main(args.reference, args.input, args.output, args.pred_keys, args.gt_key)

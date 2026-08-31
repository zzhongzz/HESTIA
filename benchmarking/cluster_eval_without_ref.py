#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def _compute_CHAOS(clusterlabel, location):
    clusterlabel = np.array(clusterlabel)
    location = np.array(location)
    matched_location = StandardScaler().fit_transform(location)
    clusterlabel_unique = np.unique(clusterlabel)
    dist_val = np.zeros(len(clusterlabel_unique))
    count = 0
    for k in clusterlabel_unique:
        location_cluster = matched_location[clusterlabel == k, :]
        if len(location_cluster) <= 2:
            continue
        nbrs = NearestNeighbors(n_neighbors=1).fit(location_cluster)
        distances, _ = nbrs.kneighbors()  # if use nbrs.kneighbors(location_cluster), then set n_neighbors=2
        dist_val[count] = np.sum(distances)
        count = count + 1
    return np.sum(dist_val) / len(clusterlabel)


def _compute_PAS(clusterlabel, location, k=10):
    clusterlabel = np.array(clusterlabel)
    location = np.array(location)
    matched_location = np.array(location)
    # matched_location = StandardScaler().fit_transform(location)
    nbrs = NearestNeighbors(n_neighbors=k).fit(matched_location)
    indices = nbrs.kneighbors(return_distance=False)
    num_diff_label = np.count_nonzero(clusterlabel[indices] != clusterlabel[..., None], axis=1)
    results = np.count_nonzero(num_diff_label > k / 2)
    return results / len(clusterlabel)


def compute_CHAOS(adata, pred_key, spatial_key="spatial"):
    return _compute_CHAOS(adata.obs[pred_key].to_numpy(), adata.obsm[spatial_key])


def compute_PAS(adata, pred_key, spatial_key="spatial"):
    return _compute_PAS(adata.obs[pred_key].to_numpy(), adata.obsm[spatial_key])


def compute_ASW(adata, pred_key, spatial_key="spatial", sample_size=100000):
    asw = silhouette_score(
        X=adata.obsm[spatial_key], labels=adata.obs[pred_key].to_numpy(), random_state=42, sample_size=sample_size
    )
    # rescale from [-1, 1] to [0, 1]
    return (asw + 1) / 2


########################### main ######################################


def main(pred_path, outpath, pred_keys, sample_size):
    print("reading input...")
    pred_adata = sc.read_h5ad(pred_path)
    output_dfs = []
    for i, pred_key in enumerate(pred_keys):
        print(f"calculating metrics for {pred_key}...")
        ## Continuity
        chaos = compute_CHAOS(pred_adata, pred_key)
        print("CHAOS(down): ", chaos)
        pas = compute_PAS(pred_adata, pred_key, spatial_key="spatial")
        print("PAS(down): ", pas)
        asw = compute_ASW(pred_adata, pred_key, spatial_key="spatial", sample_size=sample_size)
        print("ASW(up): ", asw)

        ASW_colname = "ASW(up)" if sample_size is None else f"ASW(up) ({sample_size} samples)"
        output_df = pd.DataFrame()
        output_df["type"] = ["Continuity", "Continuity", "Continuity"]
        output_df["name"] = [
            "CHAOS(down)",
            "PAS(down)",
            ASW_colname,
        ]
        output_df["value"] = [chaos, pas, asw]
        output_df.insert(loc=0, column="pred_key", value=pred_key)
        output_dfs.append(output_df)
    pd.concat(output_dfs).to_csv(outpath, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input", type=str, required=True, help="Cluster file with .h5ad format")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output tsv file")
    parser.add_argument(
        "--pred_keys",
        type=str,
        required=True,
        nargs="+",
        help="Predict Cluster key in stored in anndata.obs. Multiple keys can be separated by space",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        help="Sample size for calculating silhouette score. If not provided, all bins will be used, which is slow for small bin size",
    )
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    main(
        args.input,
        args.output,
        args.pred_keys,
        args.sample_size,
    )

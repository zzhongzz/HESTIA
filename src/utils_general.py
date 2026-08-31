import hashlib
import json
import re
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from natsort import natsorted
from safetensors.torch import load_file
from sklearn.cluster import KMeans

# filter warnings in scanpy rank_genes_groups
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")


def load_safetensors(safetensors_path, key: str, precision="16-mixed"):
    embs = load_file(safetensors_path)
    features = embs[key]
    if "16" not in str(precision) and features.dtype == torch.float16:
        features = features.float()
    return features


def cluster(
    _adata,
    _suffix,
    bin_size,
    scale,
    n_clusters: Optional[int] = None,
    resolution: Optional[float] = None,
    filename_suffix="",
    umap=True,
    is_cellbin=False,
    rank_genes_groups=False,
):
    if n_clusters is not None and n_clusters > 0:
        clusters = KMeans(n_clusters, random_state=100).fit_predict(_adata.obsm[f"X{_suffix}"])
        _adata.obs[f"kmeans{_suffix}"] = pd.Categorical(
            values=clusters.astype("U"),
            categories=natsorted(map(str, np.unique(clusters))),
        )
        if rank_genes_groups:
            try:
                if scale:
                    sc.tl.rank_genes_groups(
                        _adata,
                        groupby=f"kmeans{_suffix}",
                        key_added=f"rank_genes_groups_kmeans{_suffix}",
                        method="t-test",
                        layer="log1p",
                        use_raw=False,
                    )
                else:
                    sc.tl.rank_genes_groups(
                        _adata,
                        groupby=f"kmeans{_suffix}",
                        key_added=f"rank_genes_groups_kmeans{_suffix}",
                        method="t-test",
                        use_raw=False,
                    )
            except ValueError as e:
                warnings.warn(f"rank_genes_groups for kmeans failed: {e}")
    if resolution is not None and resolution > 0:
        sc.pp.neighbors(_adata, use_rep=f"X{_suffix}", key_added=f"neighbors{_suffix}")
        sc.tl.leiden(
            _adata,
            resolution=resolution,
            key_added=f"leiden{_suffix}",
            neighbors_key=f"neighbors{_suffix}",
            flavor="igraph",  # use future default of scanpy
            n_iterations=2,  # use future default of scanpy
        )
    if umap:
        sc.tl.umap(_adata, neighbors_key=f"neighbors{_suffix}")
    plot_cluster(_adata, _suffix, bin_size, filename_suffix, umap=umap, is_cellbin=is_cellbin)
    if rank_genes_groups:
        try:
            if scale:
                sc.tl.rank_genes_groups(
                    _adata,
                    groupby=f"leiden{_suffix}",
                    key_added=f"rank_genes_groups_leiden{_suffix}",
                    method="t-test",
                    layer="log1p",
                    use_raw=False,
                )
            else:
                sc.tl.rank_genes_groups(
                    _adata,
                    groupby=f"leiden{_suffix}",
                    key_added=f"rank_genes_groups_leiden{_suffix}",
                    method="t-test",
                    use_raw=False,
                )
        except ValueError as e:
            warnings.warn(f"rank_genes_groups for leiden failed: {e}")


def plot_cluster(_adata, _suffix, bin_size, filename_suffix="", umap=True, is_cellbin=False):
    if f"leiden{_suffix}" in _adata.obs:
        sc.pl.spatial(
            _adata,
            color=[f"leiden{_suffix}"],
            spot_size=bin_size,
            show=False,
            save=f"_bin{bin_size}_leiden{_suffix}{filename_suffix}.png"
            if not is_cellbin
            else f"_cellbin_leiden{_suffix}{filename_suffix}.png",
        )
    if umap:
        sc.pl.umap(
            _adata, color=[f"leiden{_suffix}"], show=False, save=f"_bin{bin_size}_umap{_suffix}{filename_suffix}.png"
        )
    if f"kmeans{_suffix}" in _adata.obs:
        sc.pl.spatial(
            _adata,
            color=[f"kmeans{_suffix}"],
            spot_size=bin_size,
            show=False,
            save=f"_bin{bin_size}_kmeans{_suffix}{filename_suffix}.png"
            if not is_cellbin
            else f"_cellbin_kmeans{_suffix}{filename_suffix}.png",
        )


def sanitize_filename(s):
    """Convert string to valid filename by replacing invalid characters"""
    # Replace invalid characters with underscore
    s = re.sub(r'[<>:"/\\|?*]', "_", str(s))
    # Replace multiple underscores with single underscore
    s = re.sub(r"_+", "_", s)
    # Remove leading/trailing underscores
    s = s.strip("_")
    return s


def get_cache_key(args_dict):
    """Generate a unique cache key based on input arguments"""
    # Remove output directory from cache key since it shouldn't affect the results
    if "output" in args_dict:
        args_dict = args_dict.copy()
        del args_dict["output"]
    # Convert to sorted string to ensure consistent ordering
    args_str = json.dumps(args_dict)
    return hashlib.md5(args_str.encode()).hexdigest()


def get_cache_filename(base_name, args_dict, extension="safetensors"):
    """Generate a human-readable cache filename with hash"""
    # Create a descriptive part of the filename
    desc_parts = []
    for key, value in args_dict.items():
        # don't include path in filename
        if "path" in key:
            continue
        if isinstance(value, (str, int, float, bool)):
            # Convert boolean to 0/1 for brevity
            if isinstance(value, bool):
                value = int(value)
            desc_parts.append(f"{key}_{value}")

    # Join parts and sanitize
    desc = sanitize_filename("_".join(desc_parts))
    # Truncate description if too long
    if len(desc) > 100:
        desc = desc[:97] + "..."

    # Calcuate hash
    cache_key = get_cache_key(args_dict)
    # Combine with hash
    return f"{base_name}_{desc}_{cache_key}.{extension}"


def standardize_tensor(tensor: torch.Tensor):
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, keepdim=True, unbiased=False)
    standardized_tensor = (tensor - mean) / std
    return standardized_tensor


def interaction(tensor1: torch.Tensor, tensor2: torch.Tensor, pca_dim: int):
    interaction = tensor1[:, :, None] * tensor2[:, None, :]
    interaction = interaction.reshape(interaction.shape[0], -1)
    interaction = torch.matmul(interaction, torch.pca_lowrank(interaction, q=pca_dim)[2])
    interaction = standardize_tensor(interaction)
    return interaction

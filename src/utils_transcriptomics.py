import logging
from copy import deepcopy
from typing import Literal

import numpy as np
import pandas as pd
import scanpy as sc
from numpy.typing import NDArray
from scanpy import AnnData
from scipy.sparse import csr_matrix, issparse


def get_raw_counts(adata: AnnData, layer=None, inplace=True):
    """Extracts raw count data from an AnnData object with the following order:
    1. If `layer` is not None, use `layer` as the source of raw counts.
    2. If `layer` is None, check if `.raw.X` is available and use it.
    3. If `layer` is None and `.raw.X` is not available, use `.X` as the source of raw counts.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object
    layer : str, optional
        The name of the layer to extract raw counts from, by default None
    inplace : bool, optional
        If True, replace the input `adata.X` with raw counts inplace, by default True

    Raises
    ------
    ValueError
        If the count matrix contains decimal or negative values, indicating that
        it is not raw counts.

    Returns
    -------
    numpy.ndarray | scipy.sparse.spmatrix
        The raw count matrix
    """

    def check_counts(X):
        if issparse(X):
            if (X.data % 1 != 0).any():
                raise ValueError("The count matrix contain decimals, it should be raw counts.")
            if (X.data < 0).any():
                raise ValueError("The count matrix contain negative values, it should be raw counts.")
        else:
            if (X % 1 != 0).any():
                raise ValueError("The count matrix contain decimals, it should be raw counts.")
            if (X < 0).any():
                raise ValueError("The count matrix contain negative values, it should be raw counts.")

    adata_ = adata.copy() if not inplace else adata
    if layer is not None:
        if layer not in adata_.layers:
            raise KeyError(f"layer '{layer}' not found in adata.layers")
        check_counts(adata_.layers[layer])
        adata_.X = adata_.layers[layer].copy()
        logging.info(f"using .layers['{layer}'] as raw counts")
    elif adata_.raw is not None:
        try:
            check_counts(adata_.raw.X)
            adata_.X = adata_.raw.X.copy()  # .raw.X may has different shape with .X
            logging.info("using .raw.X as raw counts")
        except Exception:
            logging.debug("failed to use .raw.X, using .X instead.", exc_info=True)
            check_counts(adata_.X)
            logging.info("using .X as raw counts")
    else:
        check_counts(adata_.X)
        logging.info("using .X as raw counts")

    return adata_.X


def scanpy_preprocess(
    adata: sc.AnnData,
    exp_mode: Literal["pca", "hvg", None],
    scale=True,
    n_pcs=50,
    n_hvg=3000,
    check_raw_counts=True,
    min_genes=1,
    min_cells=1,
    verbose=False,
):
    if adata.uns["bin_type"] == "bins":
        print(f"preprocessing h5ad of bin{adata.uns['bin_size']}...")
    elif adata.uns["bin_type"] == "cell_bins":
        print("preprocessing h5ad of cellbin...")
    else:
        raise ValueError(f"Unrecognized bin type {adata.uns['bin_type']}")
    if verbose:
        print(f"filtering cells with at least {min_genes} genes...")
    sc.pp.filter_cells(adata, min_genes=min_genes)
    if verbose:
        print(f"filtering genes which are expressed in at least {min_cells} cells...")
    sc.pp.filter_genes(adata, min_cells=min_cells)
    # Saving count data
    if verbose:
        print("saving count data...")
    if check_raw_counts:
        get_raw_counts(adata, inplace=True)
    else:
        try:
            get_raw_counts(adata, inplace=True)
        except Exception:
            logging.warning("Raw counts cannot be obtained, using .X by default")
    adata.layers["counts"] = adata.X.copy()
    # Normalizing to median total counts
    if verbose:
        print("normalizing to median total counts...")
    sc.pp.normalize_total(adata)
    # Logarithmize the data
    if verbose:
        print("logarithmizing the data...")
    sc.pp.log1p(adata)
    if scale:
        # Saving log1p data
        adata.layers["log1p"] = adata.X.copy()
    if exp_mode == "pca":
        if scale:
            if verbose:
                print("scaling the data...")
            sc.pp.scale(adata, zero_center=False)
        if verbose:
            print(f"running PCA with {n_pcs} components...")
        sc.pp.pca(adata, n_comps=n_pcs, mask_var=None, svd_solver="arpack")
    elif exp_mode == "hvg":
        if verbose:
            print(f"finding {n_hvg} highly variable genes...")
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg)
    elif exp_mode is None:
        pass
    else:
        raise NotImplementedError(f"expression processing method '{exp_mode}' is not implemented")
    return adata


def high2low(high: NDArray, lowres_binsize: int):
    return (high // lowres_binsize) * lowres_binsize


def low2high(low: NDArray, highres_binsize: int, binsize_ratio: int):
    y, x = np.indices((binsize_ratio, binsize_ratio))
    offsets_xy = np.column_stack(
        (x.ravel() * highres_binsize, y.ravel() * highres_binsize)
    )  # shape is (binsize_ratio**2, 2)
    return low[:, None, :] + offsets_xy[None, :, :]  # shape is (n_lowres_bin, binsize_ratio**2, 2)


def highres_adata_to_lowres(highres_adata: sc.AnnData, lowres_binsize: int, layer=None):
    if lowres_binsize % highres_adata.uns["bin_size"] > 0:
        print(
            f"Warning: lowres binsize {lowres_binsize} is not divisible by highres binsize {highres_adata.uns['bin_size']}, which may introduce noise"
        )
    lowres_xys = high2low(highres_adata.obsm["spatial"], lowres_binsize)
    unique_xys, inverse_indices = np.unique(lowres_xys, axis=0, return_inverse=True)

    # summed at inverse_indices instead of using for loop
    if layer is not None:
        data = highres_adata.layers[layer]
    else:
        data = highres_adata.X
    if issparse(data):
        # create a sparse matrix with ones to aggregate the data
        aggregator = csr_matrix(
            (
                np.ones_like(inverse_indices),  # data
                (
                    inverse_indices,  # row
                    np.arange(len(inverse_indices)),  # col
                ),
            ),
            shape=(len(unique_xys), len(inverse_indices)),
        )  # shape is (lowres_bins, highres_bins)
        # (lowres_bins, highres_bins) @ (highres_bins, n_features) = (lowres_bins, n_features)
        summed_matrix = aggregator @ data
    else:
        # for dense matrix, np.add.at is more straightforward and don't need to create a aggregator matrix
        summed_matrix = np.zeros((len(unique_xys), data.shape[1]), dtype=data.dtype)
        np.add.at(summed_matrix, inverse_indices, data)

    # sort by x then y
    sort_order = np.lexsort((unique_xys[:, 1], unique_xys[:, 0]))
    sorted_xys = unique_xys[sort_order]
    sorted_sums = summed_matrix[sort_order]

    lowres_X = sorted_sums
    lowres_obs = pd.DataFrame(sorted_xys, columns=["x", "y"])
    lowres_obsm = {"spatial": lowres_obs.to_numpy()}

    lowres_uns = deepcopy(highres_adata.uns)
    lowres_uns["bin_type"] = "bins"  # lowres is square bin
    lowres_uns["bin_size"] = lowres_binsize
    # Clear all values in key_record
    if "key_record" in lowres_uns:
        lowres_uns["key_record"] = {key: np.array([], dtype=np.float64) for key in lowres_uns["key_record"]}
        # Remove any keys from key_record that might be present in root level
        for key in lowres_uns["key_record"]:
            lowres_uns.pop(key, None)
    # Remove log1p if it exists
    lowres_uns.pop("log1p", None)

    lowres_var = highres_adata.var
    return sc.AnnData(X=lowres_X, obs=lowres_obs, var=lowres_var, uns=lowres_uns, obsm=lowres_obsm)

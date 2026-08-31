import os
from pathlib import Path
from typing import Optional

import anndata
import cv2 as cv
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from safetensors.torch import load_file
from skimage.io import imread
from skimage.util import img_as_ubyte
from torch import Tensor


class Preprocess:
    def __init__(self, tissue_bin_h5ad, img_regist_path: Path, bin_size: int, tile_size=4096, use_full_image=False):
        if isinstance(tissue_bin_h5ad, (str, os.PathLike)):
            self.tissue_bin_h5ad = anndata.read_h5ad(tissue_bin_h5ad)
        else:
            self.tissue_bin_h5ad = tissue_bin_h5ad
        self.img_regist_path = img_regist_path
        self.bin_size = bin_size
        self.tile_size = tile_size
        self.use_full_image = use_full_image
        self.img_tissue_bin_pad: Optional[NDArray] = None
        self.tissue_bin_coords: Optional[NDArray] = None  # x, y. Coordinates on original registered image
        self.tissue_bin_coords_on_tissue_bin_img: Optional[NDArray] = None  # x, y. Coordinates on tissue bin image
        self.min_row: Optional[int] = None  # start position of tissue bin
        self.max_row: Optional[int] = None  # end position of tissue bin, not padded
        self.min_col: Optional[int] = None
        self.max_col: Optional[int] = None

    def get_tissue_bin_coords(self):
        # get the tissue region coordinates (x, y) with a resolution of bin size
        if self.tissue_bin_coords is None:
            self.tissue_bin_coords = np.array([self.tissue_bin_h5ad.obs.x, self.tissue_bin_h5ad.obs.y]).T
        return self.tissue_bin_coords

    def get_tissue_bin_coords_on_tissue_bin_img(self):
        if self.tissue_bin_coords_on_tissue_bin_img is None:
            tissue_bin_coords = self.get_tissue_bin_coords()
            new_xs = tissue_bin_coords[:, 0] - self.min_col
            new_ys = tissue_bin_coords[:, 1] - self.min_row
            self.tissue_bin_coords_on_tissue_bin_img = np.vstack([new_xs, new_ys]).T
        return self.tissue_bin_coords_on_tissue_bin_img

    def crop_by_tissue_bin_pad(self):
        # crop the image by tissue binding box (with a resolution of bin size) and pad it to make it divisible by tile_size
        img = imread(self.img_regist_path)
        img = img_as_ubyte(img)

        if not self.use_full_image:
            tissue_bin_coords = self.get_tissue_bin_coords()
            self.min_row = int(tissue_bin_coords[:, 1].min())
            self.max_row = int(tissue_bin_coords[:, 1].max() + self.bin_size)
            self.min_col = int(tissue_bin_coords[:, 0].min())
            self.max_col = int(tissue_bin_coords[:, 0].max() + self.bin_size)
        else:
            self.min_row = 0
            self.max_row = img.shape[0]
            self.min_col = 0
            self.max_col = img.shape[1]

        img_tissue_bin_pad = img[self.min_row : self.max_row, self.min_col : self.max_col]

        # pad the image to make it divisible by tile_size
        remainder = np.array([(self.max_row - self.min_row), (self.max_col - self.min_col)]) % self.tile_size
        if (remainder != 0).any():
            complement = self.tile_size - remainder
            _pad = np.zeros((3, 2), dtype=int)
            _pad[0, 1] = complement[0]
            _pad[1, 1] = complement[1]
            self.img_tissue_bin_pad = np.pad(img_tissue_bin_pad, pad_width=_pad, mode="constant", constant_values=255)
        else:
            self.img_tissue_bin_pad = img_tissue_bin_pad
        return self.img_tissue_bin_pad


def map_img_feature_to_bin(bins_xy: NDArray, bin_size: int, bbox, token_size=16) -> pd.DataFrame:
    """Map image feature to bin position.
    Assume image feature shape is H x W x C, where H is `(image height / token_size)`, W is `(image width / token_size)`.

    ### Note:
        You can use image preprocessed with larger bin size (e.g. 256), and set the `bbox` to be that generated from the larger bin size, \
        but set `bin_size` smaller (e.g. 16) and set `bins_xy` to be that generated from the smaller bin size. \
        In this way, you don't need to run extract_features for both large and small bin size.

    Args:
        bins_xy (NDArray[int]): NDArray with shape N x 2. Where N is the number of bins under tissue. \
        Each row is the X Y position (top left corner) of a bin. \
        The bin size of this array should match with `bin_size`.
        bin_size (int): Bin size to map. This bin size can be smaller than the bin size of `bbox`, see Note for more details.
        bbox (pd.DataFrame | Iterable): Tissue bin bounding box of image used for feature extraction. \
        The 4 elements should be `min_row`, `min_col`, `max_row` and `max_col`.
        token_size (int, optional): Model token size. Defaults to 16.

    Returns:
        pd.DataFrame: Dataframe with N rows and column names `x`, `y`, `i_start`, `i_end`, `j_start`, `j_end`. \
        Each row corresponds to the X Y positon of each bin and the slice to take from the image feature to represent that bin.
    """
    if isinstance(bbox, pd.DataFrame):
        min_row, min_col, max_row, max_col = bbox.iloc[0]
    else:
        min_row, min_col, max_row, max_col = bbox
    bins_ji_offseted = (bins_xy - np.array([min_col, min_row])) / token_size
    tokens_per_bin = bin_size / token_size

    j = bins_ji_offseted[:, 0]
    i = bins_ji_offseted[:, 1]
    i_start = np.maximum(0, np.ceil(i).astype(int))
    i_end = np.ceil(i + tokens_per_bin).astype(int)
    j_start = np.maximum(0, np.ceil(j).astype(int))
    j_end = np.ceil(j + tokens_per_bin).astype(int)

    feature_slices_df = pd.DataFrame({"i_start": i_start, "i_end": i_end, "j_start": j_start, "j_end": j_end})
    bins_xy_df = pd.DataFrame(bins_xy, columns=["x", "y"])
    return pd.concat([bins_xy_df, feature_slices_df], axis=1)


def get_bounds_from_cell_border(cell_border: NDArray, centroids: NDArray):
    """Get the the minimum bounding region of cellbin from cell border and centroid

    Parameters
    ----------
    cell_border : NDArray
        NDArray with shape N x 32 x 2. Where N is the number of cellbins. \
        Each row is the X Y position of 32 vertices of each cellbin polygon. \
        If a cellbin polygon has less than 32 vertices, the X Y values for the unused vertices are 32767. \
        Noted that in SAW output, the Y values are opposite.
    centroids : NDArray
        NDArray with shape N x 2. Where N is the number of cellbins. \
        Each row is the X Y position of the centroid of a cellbin.
        
    Returns
    -------
    NDArray
        NDArray with shape N x 4. Where N is the number of cellbins. \
        Each row is the minimum bounding region (minX, minY, maxX, maxY) of a cellbin.
    """

    def get_bounds(_polygons):
        # minx, miny, maxx, maxy
        return np.concatenate([np.min(_polygons, axis=0), np.max(_polygons, axis=0)])

    mask = np.any(cell_border != 32767, axis=2)  # (n_polygons, 32) True is valid vertex
    lengths = mask.sum(axis=1)  # number of valid vertices for each polygon
    polygons = [
        cell_border[i, : lengths[i]] * np.array([1, -1]) for i in range(cell_border.shape[0])
    ]  # flip vertically to match the real shape

    # polygons: List[np.ndarray], shape of each element is (num_vertices_i, 2)
    polygons_shifted = [poly + centroids[i] for i, poly in enumerate(polygons)]

    bounds = [get_bounds(p) for p in polygons_shifted]
    return np.array(bounds)


def map_img_feature_to_cellbin(cell_bounds: NDArray, bbox, token_size=16) -> pd.DataFrame:
    """Map image feature to bin position.
    Assume image feature shape is H x W x C, where H is `(image height / token_size)`, W is `(image width / token_size)`.

    ### Note:
        You can use image preprocessed with larger bin size (e.g. 256), and set the `bbox` to be that generated from the larger bin size, \
        set `cell_bounds` to be that generated from the smaller cellbin. \
        In this way, you don't need to run extract_features for both large and small bin size (or cellbin).

    Args:
        cell_bounds (NDArray[int]): NDArray with shape N x 4. Where N is the number of cellbins. \
        Each row is the minimum bounding region (minX, minY, maxX, maxY) of a cellbin.
        bbox (pd.DataFrame | Iterable): Tissue bin bounding box of image used for feature extraction. \
        The 4 elements should be `min_row`, `min_col`, `max_row` and `max_col`.
        token_size (int, optional): Model token size. Defaults to 16.

    Returns:
        pd.DataFrame: Dataframe with N rows and column names `i_start`, `i_end`, `j_start`, `j_end`. \
        Each row corresponds to the slice to take from the image feature to represent that cellbin.
    """
    if isinstance(bbox, pd.DataFrame):
        min_row, min_col, max_row, max_col = bbox.iloc[0]
    else:
        min_row, min_col, max_row, max_col = bbox
    cell_bounds_offseted = cell_bounds.copy()
    cell_bounds_offseted[:, [0, 2]] -= min_col
    cell_bounds_offseted[:, [1, 3]] -= min_row
    cell_bounds_offseted = cell_bounds_offseted / token_size

    i_start = np.maximum(0, np.floor(cell_bounds_offseted[:, 1]).astype(int))
    i_end = np.ceil(cell_bounds_offseted[:, 3]).astype(int)
    j_start = np.maximum(0, np.floor(cell_bounds_offseted[:, 0]).astype(int))
    j_end = np.ceil(cell_bounds_offseted[:, 2]).astype(int)

    feature_slices_df = pd.DataFrame({"i_start": i_start, "i_end": i_end, "j_start": j_start, "j_end": j_end})
    return feature_slices_df


def pool_img_feature_by_bin(img_feature_to_bin: pd.DataFrame, embs: Tensor, reduction="mean"):
    """_summary_

    Args:
        img_feature_to_bin (pd.DataFrame): DataFrame generated by map_img_feature_to_bin fucntion
        embs (Tensor): Tensor of shape (H, W, C)
        reduction (str, optional): Method to reduce the dimension of embs. If None, no reduction will be made and the output shape is (N, h, w, C). Defaults to "mean".
        reduction=None only works for bin size <=64.

    Returns:
        NDArray: If reduction is not None: NDArray of shape (N, C), where N is the number of bins.
        If reduction is None: NDArray of shape (N, h, w, C), where h and w are the height and width of the bin (measured in token_size).
    """
    i_starts = img_feature_to_bin["i_start"].values
    i_ends = img_feature_to_bin["i_end"].values
    j_starts = img_feature_to_bin["j_start"].values
    j_ends = img_feature_to_bin["j_end"].values

    sub_bin_pools = []
    for i_start, i_end, j_start, j_end in zip(i_starts, i_ends, j_starts, j_ends):
        sub_bin = embs[i_start:i_end, j_start:j_end]
        if reduction == "mean":
            sub_bin_pool = sub_bin.mean(dim=(0, 1))
        elif reduction == "sum":
            sub_bin_pool = sub_bin.sum(dim=(0, 1))
        elif reduction == "max":
            sub_bin_pool = sub_bin.amax(dim=(0, 1))
        elif reduction is None:
            sub_bin_pool = sub_bin
        if np.isnan(sub_bin_pool).any():
            raise ValueError("sub_bin_pool contains NaN values")
        sub_bin_pools.append(sub_bin_pool)
    return np.array(sub_bin_pools)


def read_img_features(feature):
    if isinstance(feature, (str, os.PathLike)):
        features = load_file(feature)
    else:
        features = feature
    if all(k in features for k in ("cls", "sub")):
        embs = torch.concat([features["cls"], features["sub"]]).permute(1, 2, 0)
    else:
        raise KeyError("feature safetensors file does not contain valid keys")
    return embs


def smoothen(x, size):
    x = x.cpu().numpy()
    kernel = np.ones((size, size), np.float32) / size**2
    smoothened = cv.filter2D(x, ddepth=-1, kernel=kernel, borderType=cv.BORDER_REFLECT)
    if smoothened.ndim == 2:
        smoothened = smoothened[..., None]
    return torch.tensor(smoothened)


def smoothen_embeddings_img(embs, size):
    out = {}
    for g, emb in embs.items():
        smoothened = [smoothen(c[..., None], size=size)[..., 0] for c in emb]
        smoothened = torch.stack(smoothened)
        out[g] = smoothened
    return out

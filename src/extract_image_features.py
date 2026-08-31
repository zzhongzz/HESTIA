from itertools import product
from pathlib import Path

import numpy as np
import torch
from einops import rearrange, repeat
from numpy.typing import NDArray
from tqdm import tqdm, trange

from HIPT_4K.hipt_4k import HIPT_4K
from HIPT_4K.hipt_model_utils import eval_transforms


def tiling(img: NDArray[np.uint8], tile_size=4096):
    original_shape = np.array(img.shape[:2])
    padded_shape = (original_shape + tile_size - 1) // tile_size * tile_size
    img = np.pad(
        img,
        ((0, padded_shape[0] - img.shape[0]), (0, padded_shape[1] - img.shape[1]), (0, 0)),
        mode="constant",
        constant_values=255,
    )
    tile_shape = np.array(img.shape[:2]) // tile_size
    tile_indexes = list(product(range(tile_shape[0]), range(tile_shape[1])))
    tiles = []
    for tile_index in tile_indexes:
        row_start = tile_index[0] * tile_size
        row_end = row_start + tile_size
        col_start = tile_index[1] * tile_size
        col_end = col_start + tile_size
        tiles.append(img[row_start:row_end, col_start:col_end])

    return tiles, original_shape, tile_shape


@torch.no_grad()
def latent_256(model: HIPT_4K, x: NDArray[np.uint8]):
    x = eval_transforms()(x)  # [c x 4096 x 4096] for 4096x4096 image tile
    cls256, sub256 = model.forward_all256(x[None])
    cls256 = cls256[0].permute(1, 2, 0)  # [h_256 x w_256 x 384], where w_256=h_256=16 for 4096x4096 image tile
    sub256 = sub256[0].permute(1, 2, 3, 4, 0)  # [h_256 x w_256 x 16 x 16 x 384]
    return cls256.cpu(), sub256.cpu()


@torch.no_grad()
def latent_4k(model: HIPT_4K, x: torch.Tensor) -> torch.Tensor:
    x = x.permute(2, 0, 1)  # [384 x (h_tile x h_256) x (w_tile x w_256)]
    cls4k, sub4k = model.forward_all4k(x[None])
    sub4k = sub4k[0].permute(1, 2, 0)  # [(h_tile x h_256) x (w_tile x w_256) x 192]
    return sub4k.cpu()


def latent_256_from_tiles(model: HIPT_4K, tiles: NDArray[np.uint8]):
    sub256_tiles = []
    cls256_tiles = []
    for i in trange(len(tiles), desc="tile", leave=False):
        cls256, sub256 = latent_256(model, tiles[i])
        cls256_tiles.append(cls256)
        sub256_tiles.append(sub256)

    sub256_tiles = torch.stack(sub256_tiles)  # [n_tiles x h_256 x w_256 x 16 x 16 x 384]
    cls256_tiles = torch.stack(cls256_tiles)  # [n_tiles x h_256 x w_256 x 384]
    return sub256_tiles, cls256_tiles


def extract_embeddings_from_img(img, device="cuda"):
    tiles, _original_shape, tile_shape = tiling(img)

    model256_path = f"{Path(__file__).parent}/HIPT_4K/Checkpoints/vit256_small_dino.pth"
    model4k_path = f"{Path(__file__).parent}/HIPT_4K/Checkpoints/vit4k_xs_dino.pth"
    model = HIPT_4K(model256_path=model256_path, model4k_path=model4k_path, device256=device, device4k=device)
    model.eval()

    sub256_tiles, cls256_tiles = latent_256_from_tiles(model, tiles)

    cls256_tiles = rearrange(
        cls256_tiles, "(h1 w1) h2 w2 c -> (h1 h2) (w1 w2) c", h1=tile_shape[0], w1=tile_shape[1]
    )  # [(h_tile x h_256) x (w_tile x w_256) x 384]

    sub4k = latent_4k(model, cls256_tiles)  # [(h_tile x h_256) x (w_tile x w_256) x 192]
    del cls256_tiles, model

    patch_size = (256, 256)
    token_size = (16, 16)
    n_tokens = tuple(a // b for a, b in zip(patch_size, token_size))
    original_shape = np.array(_original_shape) // token_size

    sub = rearrange(
        sub256_tiles,
        "(h1 w1) h2 w2 h3 w3 c -> c (h1 h2 h3) (w1 w2 w3)",
        h1=tile_shape[0],
        w1=tile_shape[1],
    )  # [384 x (h_tile x h_256 x 16) x (w_tile x w_256 x 16)]
    sub = sub[:, : original_shape[0], : original_shape[1]]

    cls = repeat(
        sub4k,
        "h12 w12 c -> c (h12 h3) (w12 w3)",
        h3=n_tokens[0],
        w3=n_tokens[1],
    )  # [192 x (h_tile x h_256 x 16) x (w_tile x w_256 x 16)]
    cls = cls[:, : original_shape[0], : original_shape[1]]

    return cls, sub


def get_embeddings_shift(img, margin=256, stride=64, device="cuda", token_size=16):
    """Shift-averaged HIPT feature extraction.

    HIPT is patch16 (subpatch=16), so `token_size` is fixed at 16.
    Returns (cls_shift, sub_shift), both on the token_size pixel grid.
    """
    assert token_size == 16, "HIPT is patch16 (subpatch=16), token_size must be 16"
    assert margin % token_size == 0, f"margin={margin} must be a multiple of token_size={token_size}"
    assert stride % token_size == 0, f"stride={stride} must be a multiple of token_size={token_size}"
    return extract_img_features(img, margin=margin, stride=stride, device=device)


def extract_img_features(img, margin=256, stride=64, device="cuda"):
    """Extract image features with shifting.

    Parameters
    ----------
    img : NumPy array
        Input image of shape (H, W, C)
    margin : int, optional
        shift margin, by default 256
    stride : int, optional
        shift stride, by default 64
    device : str, optional
        device to use, by default "cuda"

    Returns
    -------
    cls_shift: embeddings of (256 x 256)-sized patches, shape (192, H/16, W/16).
    sub_shift: embeddings of (16 x 16)-sized patches, shape (384, H, W).
    """
    SCALE = 16  # scaling factor between cls and sub

    # pad image on left and top before shifting
    margin_left_top = margin // 2
    img = np.pad(
        img,
        ((margin_left_top, 0), (margin_left_top, 0), (0, 0)),
        mode="constant",
        constant_values=255,
    )
    s = np.array(img.shape[:2]) // SCALE
    cls_shift = torch.zeros([192, *s], dtype=torch.float32)
    sub_shift = torch.zeros([384, *s], dtype=torch.float32)
    n = torch.zeros([1, *s], dtype=torch.float32)

    shift_row_list = list(range(0, margin + 1, stride))
    shift_col_list = list(range(0, margin + 1, stride))
    for shift_row, shift_col in tqdm(
        product(shift_row_list, shift_col_list), total=len(shift_row_list) ** 2, desc="Extracting image features"
    ):
        row_end = -margin + shift_row if -margin + shift_row != 0 else None
        col_end = -margin + shift_col if -margin + shift_col != 0 else None
        # print(f"shift {shift_row - margin_left_top}, {shift_col - margin_left_top}")
        im = img[shift_row:row_end, shift_col:col_end]
        cls, sub = extract_embeddings_from_img(im, device=device)
        del im
        row_start_idx, col_start_idx = shift_row // SCALE, shift_col // SCALE
        row_end_idx = row_end // SCALE if row_end is not None else None
        col_end_idx = col_end // SCALE if col_end is not None else None
        cls_shift[:, row_start_idx:row_end_idx, col_start_idx:col_end_idx] += cls.cpu()
        sub_shift[:, row_start_idx:row_end_idx, col_start_idx:col_end_idx] += sub.cpu()
        n[:, row_start_idx:row_end_idx, col_start_idx:col_end_idx] += 1
        del sub

    # average the embeddings of overlapped windows
    cls_shift /= n  # [192 x (H/256) x (W/256)]
    cls_shift[torch.isnan(cls_shift)] = 0
    sub_shift /= n  # [384 x (H/256) x (W/256)]
    sub_shift[torch.isnan(sub_shift)] = 0
    # remove the values in margin_left_top
    emb_margin_left_top = margin_left_top // SCALE
    cls_shift = cls_shift[:, emb_margin_left_top:, emb_margin_left_top:]
    sub_shift = sub_shift[:, emb_margin_left_top:, emb_margin_left_top:]

    return cls_shift, sub_shift

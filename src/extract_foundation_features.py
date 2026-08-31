import json
import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from einops import rearrange, repeat
from numpy.typing import NDArray
from PIL import Image
from tqdm import tqdm

# Shared foundation ViT params (uni/gigapath are patch16, 224 input, imagenet normalization)
PATCH = 224            # input patch size in pixels
MODEL_INPUT = 224      # model input size
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def get_eval_transform():
    """ToTensor + Normalize directly on an exact 224x224 PIL patch, no resize/center-crop."""
    from torchvision import transforms

    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=_MEAN, std=_STD)])


@dataclass
class ModelConfig:
    name: str
    extra_kwargs: dict = None           # official timm_kwargs layered on top of local-dir (completes incomplete config)
    tile_embed: Callable = None         # tile_embed(model, x, n_prefix) -> [B, D_tile]: patch-level embedding (source of cls)

    def __post_init__(self):
        if self.extra_kwargs is None:
            self.extra_kwargs = {}
        if self.tile_embed is None:
            self.tile_embed = _tile_embed_default


def _tile_embed_default(model, x, n_prefix):
    """uni/uni2-h/gigapath: timm global_pool='token' -> model(x) directly gives [B, D]."""
    return model(x)


def _tile_embed_virchow2(model, x, n_prefix):
    """Virchow2: global_pool='' -> model(x)=[B, N, D]; per official convention take cat(CLS, patch.mean)=[B, 2D].
    token 0 = CLS, 1:n_prefix = reg tokens, n_prefix: = patch tokens.
    """
    out = model(x)  # [B, N, D]
    cls = out[:, 0]
    patch_mean = out[:, n_prefix:].mean(dim=1)
    return torch.cat([cls, patch_mean], dim=-1)  # [B, 2D]


def _kwargs_uni():
    """UNI official timm_kwargs (ViT-L/16, layerscale, D=1024)."""
    return dict(
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True,
    )


def _kwargs_uni2h():
    """UNI2-h official timm_kwargs (ViT-giant/14, SwiGLU, reg_tokens=8, D=1536)."""
    import timm
    import torch as _torch

    return dict(
        img_size=224,
        patch_size=14,
        depth=24,
        num_heads=24,
        init_values=1e-5,
        embed_dim=1536,
        mlp_ratio=2.66667 * 2,
        num_classes=0,
        no_embed_class=True,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=_torch.nn.SiLU,
        reg_tokens=8,
        dynamic_img_size=True,
    )


def _kwargs_virchow2():
    """Virchow2 needs mlp_layer/act_layer added (config.json omits SwiGLU).
    The rest of the arch fields (mlp_ratio=5.3375, init_values=1e-5, reg_tokens=4, dynamic_img_size) are in config.
    """
    import timm
    import torch as _torch

    return dict(
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=_torch.nn.SiLU,
    )


def _build(model_dir: Path, ckpt: Path, extra_kwargs: dict):
    """timm local-dir: + checkpoint_path, optionally layering extra_kwargs to complete arch."""
    import timm

    return timm.create_model(
        f"local-dir:{model_dir}",
        pretrained=False,
        checkpoint_path=str(ckpt),
        **extra_kwargs,
    )


MODEL_CONFIGS = {
    "uni": ModelConfig(
        name="uni",
        extra_kwargs=_kwargs_uni(),
    ),
    "uni2-h": ModelConfig(
        name="uni2-h",
        extra_kwargs=_kwargs_uni2h(),
    ),
    "gigapath": ModelConfig(
        name="gigapath",
        # config.json already contains complete model_args, no layering needed
    ),
    "virchow2": ModelConfig(
        name="virchow2",
        extra_kwargs=_kwargs_virchow2(),  # add SwiGLU (config.json omits mlp_layer/act_layer)
        tile_embed=_tile_embed_virchow2,  # global_pool='' -> cat(CLS, patch.mean)
    ),
}


def _resolve_ckpt(cfg: ModelConfig, ckpt_path: Optional[str]) -> Path:
    if not ckpt_path:
        raise FileNotFoundError(
            f"[{cfg.name}] --ckpt_path is required for foundation models; "
            f"pass the local pytorch_model.bin (config.json must sit next to it)."
        )
    p = Path(ckpt_path)
    if not p.is_file():
        raise FileNotFoundError(f"[{cfg.name}] checkpoint not found: {p}")
    return p


def _probe_arch(model, device):
    probe = torch.zeros(1, 3, MODEL_INPUT, MODEL_INPUT, device=device)
    with torch.no_grad():
        toks = model.forward_features(probe)  # [1, n_total, D]
    n_total, D = toks.shape[1], toks.shape[2]

    grid = None
    num_spatial = None
    patch_size = None
    pe = getattr(model, "patch_embed", None)
    if pe is not None:
        gs = getattr(pe, "grid_size", None)
        if isinstance(gs, (tuple, list)) and len(gs) >= 2:
            grid = int(gs[0])
            num_spatial = int(gs[0]) * int(gs[1])
        np_ = getattr(pe, "num_patches", None)
        if np_ is not None:
            num_spatial = int(np_)
        ps = getattr(pe, "patch_size", None)
        if isinstance(ps, (tuple, list)):
            patch_size = int(ps[0])
        elif ps is not None:
            patch_size = int(ps)
    if grid is None or patch_size is None:
        patch_size = patch_size or 16
        grid = grid or (MODEL_INPUT // patch_size)
        num_spatial = num_spatial or (grid * grid)
    n_prefix = n_total - num_spatial
    assert n_prefix >= 0, f"n_total={n_total} < num_spatial={num_spatial}"
    print(
        f"[foundation] patch_size={patch_size} grid={grid}x{grid} embed_dim={D} "
        f"n_total={n_total} n_prefix={n_prefix} (spatial={num_spatial})"
    )
    return patch_size, grid, D, n_prefix


def get_model(model_name="uni", device="cuda", ckpt_path=None):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"unknown model {model_name!r}, choices={list(MODEL_CONFIGS)}")
    cfg = MODEL_CONFIGS[model_name]
    ckpt = _resolve_ckpt(cfg, ckpt_path)
    model_dir_resolved = ckpt.parent
    cfg_file = model_dir_resolved / "config.json"
    if not cfg_file.is_file():
        raise FileNotFoundError(
            f"[{model_name}] config.json not found next to weights: {cfg_file}. "
            f"Please place config.json in the same dir as the weights."
        )
    model = _build(model_dir_resolved, ckpt, cfg.extra_kwargs)
    print(f"[{model_name}] loaded local config+weights: {model_dir_resolved} / {ckpt.name} (offline, no HF)")
    model.eval()
    model.to(device)
    patch_size, grid, D, n_prefix = _probe_arch(model, device)
    arch = (patch_size, grid, D, n_prefix, cfg.tile_embed)
    return model, get_eval_transform(), arch


def get_token_size(model_name, ckpt_path=None):
    """Read patch_size from the model's config.json (without loading weights), for token_size auto-detection.

    `ckpt_path` (the local pytorch_model.bin) is required; config.json is read from the same dir.
    Prefer `model_args.patch_size` (gigapath's name is patch14 but model_args.patch_size=16 overrides to 16);
    otherwise parse `patch\\d+` from the `architecture` name (uni/uni2-h/virchow2 config has no patch_size in model_args).
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"unknown model {model_name!r}, choices={list(MODEL_CONFIGS)}")
    cfg = MODEL_CONFIGS[model_name]
    ckpt = _resolve_ckpt(cfg, ckpt_path)
    config_path = ckpt.parent / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"[{model_name}] config.json not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    ma = config.get("model_args") or {}
    if ma.get("patch_size") is not None:
        return int(ma["patch_size"])
    arch = config.get("architecture", "")
    m = re.search(r"patch(\d+)", arch)
    if not m:
        raise ValueError(f"[{model_name}] cannot determine patch_size from config.json: {config_path}")
    return int(m.group(1))


def _to_rgb_uint8(img: NDArray) -> NDArray[np.uint8]:
    """Ensure input is HxWx3 uint8 (drop alpha / gray->RGB)."""
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img.astype(np.uint8)


@torch.no_grad()
def get_embeddings_from_img(
    img: NDArray[np.uint8],
    model_name="uni",
    device="cuda",
    batch_size=64,
    ckpt_path=None,
    model_cache=None,
    token_size=None,
):
    img = _to_rgb_uint8(img)
    H, W = img.shape[:2]

    # pad to a multiple of PATCH (white 255, consistent with HIPT tiling)
    H_pad = (H + PATCH - 1) // PATCH * PATCH
    W_pad = (W + PATCH - 1) // PATCH * PATCH
    img_pad = np.pad(
        img,
        ((0, H_pad - H), (0, W_pad - W), (0, 0)),
        mode="constant",
        constant_values=255,
    )
    n_py, n_px = H_pad // PATCH, W_pad // PATCH

    if model_cache is not None:
        model, transform, arch = model_cache
    else:
        model, transform, arch = get_model(model_name=model_name, device=device, ckpt_path=ckpt_path)
    patch_size, grid, D, n_prefix, tile_embed = arch
    if token_size is None:
        token_size = patch_size  # default: the model's own patch_size
    assert patch_size == token_size, (
        f"model patch_size={patch_size} != token_size={token_size}: "
        f"token_size must equal the model patch_size (so the output grid aligns)"
    )
    gh, gw = H // token_size, W // token_size  # target grid, floor, aligned to (0,0)

    # generate 224 patches by row (py-major order)
    patches = []
    for py in range(n_py):
        for px in range(n_px):
            y0, x0 = py * PATCH, px * PATCH
            patches.append(Image.fromarray(img_pad[y0 : y0 + PATCH, x0 : x0 + PATCH], mode="RGB"))

    pooled_list, spatial_list = [], []
    for i in tqdm(range(0, len(patches), batch_size), desc=f"{model_name} patches", leave=False):
        batch_pil = patches[i : i + batch_size]
        batch = torch.stack([transform(p) for p in batch_pil]).to(device)
        pooled = tile_embed(model, batch, n_prefix)    # [B, D_tile] (patch-level embedding, source of cls)
        tokens = model.forward_features(batch)        # [B, n_total, D]
        spatial = tokens[:, n_prefix:, :]            # [B, grid*grid, D]  drop CLS/reg
        pooled_list.append(pooled.cpu())
        spatial_list.append(spatial.cpu())
    if model_cache is None:
        del model

    pooled = torch.cat(pooled_list, dim=0)          # [N, D_tile]
    spatial = torch.cat(spatial_list, dim=0)        # [N, grid*grid, D]
    D_tile = pooled.shape[-1]
    spatial = spatial.reshape(n_py, n_px, grid, grid, D)
    pooled = pooled.reshape(n_py, n_px, D_tile)

    # assemble into a full-image token_size px grid (n_py*grid = (H_pad/PATCH)*grid = H_pad/token_size)
    sub = rearrange(spatial, "py px gy gx D -> D (py gy) (px gx)")
    cls = repeat(pooled, "py px Dc -> Dc (py gy) (px gx)", gy=grid, gx=grid)

    sub = sub[:, :gh, :gw].contiguous()
    cls = cls[:, :gh, :gw].contiguous()
    return cls, sub


@torch.no_grad()
def get_embeddings_shift(
    img: NDArray[np.uint8],
    margin=256,
    stride=64,
    model_name="uni",
    device="cuda",
    batch_size=64,
    ckpt_path=None,
    token_size=None,
):
    # build the model only once, reuse across shifts
    model_cache = get_model(model_name=model_name, device=device, ckpt_path=ckpt_path)
    if token_size is None:
        _, _, arch = model_cache
        token_size = arch[0]  # default: the probed patch_size
    FACTOR = token_size
    assert margin % token_size == 0, f"margin={margin} must be a multiple of token_size={token_size}"
    assert stride % token_size == 0, f"stride={stride} must be a multiple of token_size={token_size}"

    margin_left_top = margin // 2
    img = np.pad(
        img,
        ((margin_left_top, 0), (margin_left_top, 0), (0, 0)),
        mode="constant",
        constant_values=255,
    )
    shape_emb = np.array(img.shape[:2]) // FACTOR

    cls_shift = None  # lazy allocation
    sub_shift = None
    n_reps = None

    start_list = list(range(0, margin + 1, stride))
    for start0, start1 in tqdm(
        product(start_list, start_list), total=len(start_list) ** 2, desc="Extracting image features"
    ):
        stop0 = -margin + start0 if -margin + start0 != 0 else None
        stop1 = -margin + start1 if -margin + start1 != 0 else None
        im = img[start0:stop0, start1:stop1]
        cls, sub = get_embeddings_from_img(
            im,
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            model_cache=model_cache,
            token_size=token_size,
        )
        del im
        sta0, sta1 = start0 // FACTOR, start1 // FACTOR
        sto0 = stop0 // FACTOR if stop0 is not None else None
        sto1 = stop1 // FACTOR if stop1 is not None else None
        if cls_shift is None:
            cls_shift = torch.zeros([cls.shape[0], *shape_emb], dtype=torch.float32)
            sub_shift = torch.zeros([sub.shape[0], *shape_emb], dtype=torch.float32)
            n_reps = torch.zeros([1, *shape_emb], dtype=torch.float32)
        cls_shift[:, sta0:sto0, sta1:sto1] += cls.cpu()
        sub_shift[:, sta0:sto0, sta1:sto1] += sub.cpu()
        n_reps[:, sta0:sto0, sta1:sto1] += 1
        del sub

    # average the embeddings of overlapped windows
    cls_shift /= n_reps
    cls_shift[torch.isnan(cls_shift)] = 0
    sub_shift /= n_reps
    sub_shift[torch.isnan(sub_shift)] = 0
    # remove the values in margin_left_top
    emb_margin_left_top = margin_left_top // FACTOR
    cls_shift = cls_shift[:, emb_margin_left_top:, emb_margin_left_top:]
    sub_shift = sub_shift[:, emb_margin_left_top:, emb_margin_left_top:]

    return cls_shift, sub_shift

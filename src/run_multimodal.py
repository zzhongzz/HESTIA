import warnings

warnings.filterwarnings(action="ignore", category=FutureWarning)
import argparse
import gc
import importlib
import os
import time
from itertools import chain
from pathlib import Path

import lightning as L
import scanpy as sc
import tifffile
import torch
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.loggers import CometLogger
from matplotlib import rcParams
from safetensors.torch import load_file, save_file
from skimage.io import imread
from skimage.util import img_as_ubyte
from torch import Tensor
from torch.utils.data import DataLoader

# Set CUBLAS workspace config for deterministic behavior
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from encode_binsize import (
    BinsizeAE,
    BinsizeAEFeatureDataset,
    HighresAE,
    HighresAEFeatureDataset,
    collate_highres_only,
    collate_highres_test,
    collate_highres_train,
)
from encode_image import ImageAE, ImgAEFeatureDataset
from utils_general import cluster, get_cache_filename, interaction, plot_cluster, standardize_tensor
from utils_image import Preprocess, smoothen_embeddings_img
from utils_transcriptomics import scanpy_preprocess

# Model dispatch: "hipt" -> HIPT two-level ViT (extract_image_features);
# "uni"/"uni2-h"/"gigapath"/"virchow2" -> single foundation ViT (extract_foundation_features).
_MODEL_MODULES = {
    "hipt": "extract_image_features",
    "uni": "extract_foundation_features",
    "uni2-h": "extract_foundation_features",
    "gigapath": "extract_foundation_features",
    "virchow2": "extract_foundation_features",
}


def _get_embeddings_shift(model, img, *, margin, stride, device, token_size, img_batch_size, ckpt_path):
    mod = importlib.import_module(_MODEL_MODULES[model])
    common = dict(margin=margin, stride=stride, device=device, token_size=token_size)
    if model == "hipt":
        return mod.get_embeddings_shift(img, **common)
    extra = dict(batch_size=img_batch_size, model_name=model)
    if ckpt_path is not None:
        extra["ckpt_path"] = ckpt_path
    return mod.get_embeddings_shift(img, **common, **extra)


def load_and_preprocess_h5ad(
    h5ad_path,
    out_dir,
    scale=True,
    n_pcs=64,
    use_cache=False,
    parameters=None,
    outfiles_name=None,
):
    """Load and preprocess h5ad file once for both binsize encoding and clustering"""
    # Generate cache key for output filename
    cache_args = {
        "h5ad_path": str(Path(h5ad_path).resolve()),
        "n_pcs": n_pcs,
        "scale": scale,
    }
    cache_filename = get_cache_filename("preprocessed", cache_args, extension="h5ad")
    cache_file = Path(out_dir).joinpath(cache_filename)
    print(cache_file)
    if use_cache and cache_file.exists():
        print(f"Loading preprocessed data from cache: {cache_file.name}")
        adata = sc.read_h5ad(cache_file)
        return adata, cache_file

    print("Loading and preprocessing h5ad file...")
    adata = sc.read_h5ad(h5ad_path)
    adata = scanpy_preprocess(
        adata, exp_mode="pca", scale=scale, n_pcs=n_pcs, check_raw_counts=False, min_cells=3, verbose=True
    )

    if use_cache:
        print(f"Saving preprocessed data to cache: {cache_file.name}")
        adata.write_h5ad(cache_file, compression="gzip")
        if parameters is not None and outfiles_name is not None:
            with open(outfiles_name, "a") as f:
                f.write(f"{cache_file.name}\t{parameters}\n")

    return adata, cache_file


def preprocess_image(
    img_path: Path,
    adata,
    out_dir: Path,
    use_full_image=False,
    bin_size=256,
    use_cache=False,
    parameters=None,
    outfiles_name=None,
):
    """Preprocess the input image by cropping to tissue region"""
    print("Preprocessing image...")

    # Generate cache key for output filenames
    # NOTE assume adata's coordinates are always the same if out_dir is the same
    cache_args = {
        "img_path": str(img_path.resolve()),
        "bin_size": bin_size,
        "use_full_image": use_full_image,
    }

    # Generate output filenames
    bbox_file = out_dir / get_cache_filename("bbox", cache_args, extension="tsv")
    img_file = out_dir.joinpath(get_cache_filename("HE", cache_args, extension="tif"))
    print(bbox_file, img_file)
    if use_cache and bbox_file.exists() and img_file.exists():
        print(f"Preprocessed files {img_file.name} and {bbox_file.name} already exist, skipping preprocessing step")
        print(
            "NOTE: we assume adata's coordinates are always the same if output directory is the same (e.g. adata with different bin size). "
            "If you use adata from different subregions, please use different output directory, otherwise you may encounter error."
        )
        return img_file, bbox_file

    # Initialize preprocessor
    preprocessor = Preprocess(
        tissue_bin_h5ad=adata,
        img_regist_path=Path(img_path),
        bin_size=bin_size,
        tile_size=bin_size,
        use_full_image=use_full_image,
    )

    # Crop image by tissue bin
    print("Cropping image by tissue bin...")
    preprocessor.crop_by_tissue_bin_pad()

    # Write output files with custom filenames
    print("Writing bbox file:", bbox_file.name)
    with open(bbox_file, "w") as f:
        f.write("min_row\tmin_col\tmax_row\tmax_col\n")
        f.write(f"{preprocessor.min_row}\t{preprocessor.min_col}\t{preprocessor.max_row}\t{preprocessor.max_col}\n")

    print("Writing preprocessed image:", img_file.name)
    tifffile.imwrite(img_file, preprocessor.img_tissue_bin_pad, compression="ZLIB")
    if parameters is not None and outfiles_name is not None:
        with open(outfiles_name, "a") as f:
            f.write(f"{bbox_file.name}\t{parameters}\n")
            f.write(f"{img_file.name}\t{parameters}\n")
    return img_file, bbox_file


def extract_he_features(
    img_path: Path,
    out_dir: Path,
    device="cuda:0",
    shift_margin=256,
    shift_stride=64,
    model="hipt",
    token_size=16,
    img_batch_size=64,
    ckpt_path=None,
    use_cache=False,
    parameters=None,
    outfiles_name=None,
):
    """Extract features from HE image"""
    print("Extracting HE image features...")
    # Generate cache key and filename for HE features
    cache_args = {
        "img_path": str(img_path.resolve()),
        "model": model,
        "token_size": token_size,
        "shift_margin": shift_margin,
    }
    if shift_margin > 0:
        cache_args["shift_stride"] = shift_stride
    cache_file = out_dir.joinpath(get_cache_filename("he_features", cache_args))
    print(cache_file)
    if use_cache and cache_file.exists():
        print("Using cached HE features from:", cache_file.name)
        return cache_file

    img = imread(img_path)
    img = img_as_ubyte(img)

    cls_shift, sub_shift = _get_embeddings_shift(
        model,
        img,
        margin=shift_margin,
        stride=shift_stride,
        device=device,
        token_size=token_size,
        img_batch_size=img_batch_size,
        ckpt_path=ckpt_path,
    )
    cls_shift = cls_shift.contiguous()
    sub_shift = sub_shift.contiguous()
    embs = dict(cls=cls_shift, sub=sub_shift)

    print("Saving HE features before smoothening to file:", cache_file.name)
    save_file(embs, cache_file)
    if parameters is not None and outfiles_name is not None:
        with open(outfiles_name, "a") as f:
            f.write(f"{cache_file.name}\t{parameters}\n")
    return cache_file


def encode_image_features(
    image_features_st_path: Path,
    bbox_path: Path,
    out_dir: Path,
    h5ad_path: Path,
    smoothen=False,
    smoothen_size=8,
    skip_encode_img=False,
    img_hidden_dim=128,
    devices=[0],
    max_epochs=1000,
    batch_size=512,
    seed=None,
    token_size=16,
    use_cache=False,
    parameters=None,
    outfiles_name=None,
):
    """Encode HE image features to match binsize"""
    print("Encoding HE image features...")

    # Generate cache key and filename for encoded HE features
    if smoothen_size <= 0:
        smoothen = False
    cache_args = {
        "img_feature_path": str(image_features_st_path.resolve()),
        "bbox_path": str(bbox_path.resolve()),
        "h5ad_path": str(h5ad_path.resolve()),
        "smoothen": smoothen,
    }
    if smoothen:
        cache_args["smoothen_size"] = smoothen_size
    if skip_encode_img:
        cache_args["skip_encode_img"] = True
    else:
        cache_args["img_hidden_dim"] = img_hidden_dim
        cache_args["max_epochs"] = max_epochs
        cache_args["batch_size"] = batch_size
        cache_args["seed"] = seed
    cache_file = out_dir.joinpath(get_cache_filename("encoded_he_features", cache_args))
    print(cache_file)
    if use_cache and cache_file.exists():
        print("Loading cached encoded HE features from:", cache_file.name)
        return load_file(cache_file)["encoded_he_features"], cache_file

    image_features_st = load_file(image_features_st_path)

    # smoothen
    if smoothen:
        print("Smoothening HE embeddings...")
        image_features_st = smoothen_embeddings_img(image_features_st, size=smoothen_size)

    # Create dataset from HE embeddings
    train_dataset = ImgAEFeatureDataset(
        h5ad=h5ad_path, bbox_path=bbox_path, image_features_st=image_features_st, token_size=token_size
    )
    if skip_encode_img:
        print("skip_encode_img is set, skipping HE feature encoding and returning raw HE features")
        encoded_he = train_dataset.image_features
    else:
        # Initialize model
        model = ImageAE(
            input_dim=train_dataset.input_dim,
            hidden_dim=img_hidden_dim,
            lr=5e-4,
            weight_decay=1e-4,
            max_epochs=max_epochs,
        )

        # Create dataloaders with minimal configuration
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            # num_workers=8,
            # pin_memory=True,
            # persistent_workers=True,
        )

        test_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            # num_workers=8,
            # pin_memory=True,
            # persistent_workers=True,
        )

        # Train model
        logger = CometLogger(
            # save_dir=out_dir.joinpath("logs"),
            # offline=True,
            # project_name="multimodal",
            # experiment_name="encoded_he",
            online=False,
            offline_directory=out_dir.joinpath("logs"),
            project="multimodal",
            name="encoded_he",
        )
        progress_bar = RichProgressBar()
        trainer = L.Trainer(
            devices=devices,
            callbacks=[progress_bar],
            max_epochs=max_epochs,
            logger=logger,
            profiler="simple",
            barebones=False,  # set to True to analyze the Trainer overhead
            deterministic=True if seed is not None else False,
        )

        trainer.fit(model, train_dataloaders=train_dataloader)
        results = trainer.predict(model, test_dataloader)

        # Process results
        encoded_he = torch.cat([batch["embeddings"] for batch in results], dim=0)
        indices = torch.cat([batch["indices"] for batch in results], dim=0)
        sorted_order = torch.argsort(indices)
        encoded_he = encoded_he[sorted_order]

    if use_cache:
        print("Saving encoded HE features to cache:", cache_file.name)
        save_file({"encoded_he_features": encoded_he}, cache_file)
        if parameters is not None and outfiles_name is not None:
            with open(outfiles_name, "a") as f:
                f.write(f"{cache_file.name}\t{parameters}\n")
    return encoded_he, cache_file


def encode_binsize_features(
    h5ad_path: Path,
    adata,
    out_dir: Path,
    binsize_ratio=4,
    devices=[0],
    max_epochs=1000,
    batch_size=512,
    seed=None,
    binsize_hidden_dim=128,
    compare_loss_weight=1.0,
    n_pcs=64,
    tie_weights=False,
    use_cache=False,
    cellbin_binsize=20,
    parameters=None,
    outfiles_name=None,
):
    """Encode binsize features"""
    print("Encoding binsize features...")

    # Generate cache key and filename for binsize features
    cache_args = {
        "h5ad_path": str(Path(h5ad_path).resolve()),
        "binsize_ratio": binsize_ratio,
    }
    if binsize_ratio >= 0:
        cache_args["binsize_hidden_dim"] = binsize_hidden_dim
        cache_args["max_epochs"] = max_epochs
        cache_args["batch_size"] = batch_size
        cache_args["tie_weights"] = tie_weights
        cache_args["seed"] = seed
        if binsize_ratio >= 1:
            cache_args["compare_loss_weight"] = compare_loss_weight

    if adata.uns["bin_type"] == "bins":
        cache_args["bin_size"] = int(adata.uns["bin_size"])  # ensure it's not int64
    elif adata.uns["bin_type"] == "cell_bins":
        cache_args["cellbin_binsize"] = cellbin_binsize
    else:
        raise ValueError(f"Unrecognized bin type {adata.uns['bin_type']}")

    cache_args["n_pcs"] = "n_pcs"
    cache_file = out_dir.joinpath(get_cache_filename("binsize_features", cache_args))
    print(cache_file)
    if use_cache and cache_file.exists():
        print("Loading cached binsize features from:", cache_file.name)
        return load_file(cache_file)["encoded_binsize_features"], cache_file
    if binsize_ratio < 0:
        print(f"{binsize_ratio=}, skipping binsize encoding and returning the PCA features directly")
        encoded_binsize_features = torch.tensor(adata.obsm["X_pca"], dtype=torch.float32).contiguous()
    else:
        if binsize_ratio >= 1:
            train_dataset = BinsizeAEFeatureDataset(
                highres_h5ad=adata,
                binsize_ratio=binsize_ratio,
                n_pcs=n_pcs,
                no_preprocess=True,
                cellbin_binsize=cellbin_binsize,
            )
            model = BinsizeAE(
                input_dim=train_dataset.input_dim,
                hidden_dim=binsize_hidden_dim,
                tie_weights=tie_weights,
                compare_loss_weight=compare_loss_weight,
                lr=5e-4,
                weight_decay=1e-4,
                max_epochs=max_epochs,
            )
        else:
            print("binsize_ratio=0, skipping binsize encoding and returning highres features only")
            train_dataset = HighresAEFeatureDataset(
                highres_h5ad=adata,
                binsize_ratio=binsize_ratio,
                n_pcs=n_pcs,
                no_preprocess=True,
                cellbin_binsize=cellbin_binsize,
            )
            model = HighresAE(
                input_dim=train_dataset.input_dim,
                hidden_dim=binsize_hidden_dim,
                tie_weights=tie_weights,
                compare_loss_weight=compare_loss_weight,
                lr=5e-4,
                weight_decay=1e-4,
                max_epochs=max_epochs,
            )

        train_dataloder = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_highres_train if binsize_ratio >= 1 else collate_highres_only,
            # num_workers=8,
            pin_memory=True,
            # persistent_workers=True,
        )

        test_dataloder = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_highres_test if binsize_ratio >= 1 else collate_highres_only,
            # num_workers=8,
            pin_memory=True,
            # persistent_workers=True,
        )
        logger = CometLogger(
            # save_dir=out_dir.joinpath("logs"),
            # offline=True,
            # project_name="multimodal",
            # experiment_name="encoded_binsize",
            online=False,
            offline_directory=out_dir.joinpath("logs"),
            project="multimodal",
            name="encoded_binsize",
        )
        progress_bar = RichProgressBar()
        trainer = L.Trainer(
            devices=devices,
            callbacks=[progress_bar],
            max_epochs=max_epochs,
            logger=logger,
            profiler="simple",
            barebones=False,  # set to True to analyze the Trainer overhead
            deterministic=True if seed is not None else False,
        )

        trainer.fit(model, train_dataloaders=train_dataloder)
        results = trainer.predict(model, test_dataloder)

        # Process results
        def sort_by_indices(data, indices):
            sorted_order = torch.argsort(indices, dim=0)
            return data[sorted_order]

        if binsize_ratio >= 1:
            highres_embs = list(chain(*chain(*[batch["highres"]["embeddings"] for batch in results])))
            highres_indices = list(chain(*chain(*[batch["highres"]["indices"] for batch in results])))
            encoded_binsize_features = sort_by_indices(torch.stack(highres_embs), torch.stack(highres_indices))
        else:
            highres_embs = torch.cat([batch["highres"]["embeddings"] for batch in results], dim=0)
            highres_indices = torch.cat([batch["highres"]["indices"] for batch in results], dim=0).squeeze()
            encoded_binsize_features = sort_by_indices(highres_embs, highres_indices)

    if use_cache:
        print("Saving binsize features to cache:", cache_file.name)
        save_file({"encoded_binsize_features": encoded_binsize_features}, cache_file)
        if parameters is not None and outfiles_name is not None:
            with open(outfiles_name, "a") as f:
                f.write(f"{cache_file.name}\t{parameters}\n")
    return encoded_binsize_features, cache_file


def combine_features(
    encoded_he_features,
    encoded_binsize_features,
    he_features_weight=1.0,
    interaction_npcs=64,
) -> dict[str, Tensor]:
    """Combine HE and binsize features through interaction"""
    print("Combining features through interaction...")

    # Weight HE features
    encoded_he_features = encoded_he_features * he_features_weight

    # Standardize embeddings
    encoded_he_features_std = standardize_tensor(encoded_he_features)
    encoded_binsize_features_std = standardize_tensor(encoded_binsize_features)

    # Create interactions
    binsize_he_interaction = interaction(encoded_binsize_features, encoded_he_features, pca_dim=interaction_npcs)

    # Combine standardized embeddings
    combined_std = torch.cat([encoded_he_features_std, encoded_binsize_features_std], dim=1)

    # Combine standardized embeddings with interactions
    combined_with_interaction = torch.cat([combined_std, binsize_he_interaction], dim=1)

    # Return all features
    return {
        "encoded_he_features": encoded_he_features,
        "encoded_binsize_features": encoded_binsize_features,
        "binsize_he_interaction": binsize_he_interaction,
        "combined_standardized": combined_std,
        "combined_with_interaction": combined_with_interaction,
    }


def cluster_features(
    h5ad_path: Path,
    combined_features: dict[str, Tensor],
    out_dir: Path,
    encoded_he_path: Path,
    encoded_binsize_path: Path,
    he_features_weight=1.0,
    interaction_npcs=64,
    included_features=[],
    n_clusters=10,
    resolution=1.0,
    scale=True,
    use_cache=False,
    cellbin_binsize=20,
    parameters=None,
    outfiles_name=None,
):  # -> tuple[Path, list]:
    # Generate cache key for output filename
    cache_args = {
        "h5ad_path": str(h5ad_path.resolve()),
        "encoded_he_path": str(encoded_he_path.resolve()),
        "encoded_binsize_path": str(encoded_binsize_path.resolve()),
        "he_features_weight": he_features_weight,
        "interaction_npcs": interaction_npcs,
        "scale": scale,
        "n_clusters": n_clusters,
        "resolution": resolution,
        "included_features": included_features,
        "cellbin_binsize": cellbin_binsize,
    }
    cache_filename = get_cache_filename("clustered", cache_args, extension="h5ad")
    cache_file = out_dir.joinpath(cache_filename)

    # Set up scanpy settings
    rcParams["figure.figsize"] = (16, 12)
    sc.settings.figdir = out_dir

    """Cluster the combined features using scanpy"""
    print("Clustering features...")
    plots = []
    print(cache_file)
    if use_cache and cache_file.exists():
        print(f"Loading clustered data from cache: {cache_file.name}")
        adata = sc.read_h5ad(cache_file)
        if adata.uns["bin_type"] == "bins":
            bin_size = adata.uns["bin_size"]
            bin_size_str = f"bin{bin_size}"
            is_cellbin = False
        elif adata.uns["bin_type"] == "cell_bins":
            bin_size = cellbin_binsize
            bin_size_str = "cellbin"
            is_cellbin = True
        else:
            raise ValueError(f"Unrecognized bin type {adata.uns['bin_type']}")

        for key in combined_features.keys():
            if key not in included_features:
                continue
            if resolution > 0 or n_clusters > 0:
                print(f"Plotting clustered results: {key}")
            suffix = f"_{key}"
            filename_suffix = f"_{cache_file.stem}"
            plot_cluster(adata, suffix, bin_size, filename_suffix=filename_suffix, umap=False, is_cellbin=is_cellbin)
            if resolution > 0:
                plots.append(f"show_{bin_size_str}_leiden{suffix}{filename_suffix}.png")
            if n_clusters > 0:
                plots.append(f"show_{bin_size_str}_kmeans{suffix}{filename_suffix}.png")
    else:
        adata = sc.read_h5ad(h5ad_path)
        if adata.uns["bin_type"] == "bins":
            bin_size = adata.uns["bin_size"]
            bin_size_str = f"bin{bin_size}"
            is_cellbin = False
        elif adata.uns["bin_type"] == "cell_bins":
            bin_size = cellbin_binsize
            bin_size_str = "cellbin"
            is_cellbin = True
        else:
            raise ValueError(f"Unrecognized bin type {adata.uns['bin_type']}")

        # Add embeddings to adata and cluster
        for key in combined_features.keys():
            if key not in included_features:
                continue
            if resolution > 0 or n_clusters > 0:
                print(f"Clustering by {key}...")
            suffix = f"_{key}"
            filename_suffix = f"_{cache_file.stem}"
            try:
                adata.obsm[f"X{suffix}"] = combined_features[key].numpy()
            except Exception as e:
                print(f"Failed to load {key} embeddings to .obsm")
                print(e)
                continue
            cluster(
                adata,
                suffix,
                bin_size,
                scale=scale,
                n_clusters=n_clusters,
                resolution=resolution,
                filename_suffix=filename_suffix,
                umap=False,
                is_cellbin=is_cellbin,
            )
            if resolution > 0:
                plots.append(f"show_{bin_size_str}_leiden{suffix}{filename_suffix}.png")
            if n_clusters > 0:
                plots.append(f"show_{bin_size_str}_kmeans{suffix}{filename_suffix}.png")

        # Save final results when not using cache
        print(f"Writing final results to {cache_file}")
        adata.write_h5ad(cache_file, compression="gzip")

    # Record outputs to outfiles if requested (single place)
    if parameters is not None and outfiles_name is not None:
        with open(outfiles_name, "a") as f:
            f.write(f"{cache_file.name}\t{parameters}\n")
            for p in plots:
                f.write(f"{p}\t{parameters}\n")

    return cache_file, plots


def main():
    parser = argparse.ArgumentParser(
        description="Process multimodal data: preprocess image, extract HE features, encode binsize, combine features, and cluster",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # General arguments
    general_group = parser.add_argument_group("General arguments")
    general_group.add_argument("-i", "--input", required=True, help="input HE image file")
    general_group.add_argument("-a", "--h5ad", required=True, help="input h5ad file")
    general_group.add_argument("-o", "--output", required=True, help="output directory")
    general_group.add_argument("--seed", type=int, help="random seed for reproducibility")
    general_group.add_argument("--use_cache", action="store_true", help="use cached results if available")
    general_group.add_argument("--devices", type=int, nargs="+", default=[0], help="CUDA device(s) to use")

    # Image feature arguments
    image_group = parser.add_argument_group("Image feature arguments")
    image_group.add_argument(
        "--model",
        default="hipt",
        choices=["hipt", "uni", "uni2-h", "gigapath", "virchow2"],
        help="image feature extractor",
    )
    image_group.add_argument(
        "--token_size",
        type=int,
        default=None,
        help="output feature grid pixel resolution, must equal the model patch_size. Default None=auto-detected: "
        "hipt=16; other models read patch_size from their config.json (uni/gigapath=16, uni2-h/virchow2=14).",
    )
    image_group.add_argument(
        "--img_batch_size",
        type=int,
        default=64,
        help="batch size for image feature inference (uni/uni2-h/gigapath/virchow2)",
    )
    image_group.add_argument(
        "--ckpt_path",
        help="local weight file pytorch_model.bin for other models (required when --model is not hipt). "
        "config.json must sit in the same dir as the weights. Ignored by hipt.",
    )
    image_group.add_argument(
        "--shift_margin",
        type=int,
        default=None,
        help="margin for HE feature extraction. Set to 0 to disable shifting. Default None=16*token_size; "
        "must be divisible by token_size",
    )
    image_group.add_argument(
        "--shift_stride",
        type=int,
        default=None,
        help="stride for HE feature extraction. Default None=4*token_size; must be divisible by token_size",
    )
    image_group.add_argument("--smoothen", action="store_true", help="apply smoothing to HE embeddings")
    image_group.add_argument("--smoothen_size", type=int, default=8, help="size to smooth HE embeddings")
    image_group.add_argument("--use_full_image", action="store_true", help="do not crop image to tissue region")

    # Encode image arguments
    encode_group = parser.add_argument_group("Encode image arguments")
    encode_group.add_argument("--img_hidden_dim", type=int, default=128, help="hidden dimension for encoded features")
    encode_group.add_argument("--max_epochs", type=int, default=200, help="max epochs for encoding")
    encode_group.add_argument("--batch_size", type=int, default=512, help="batch size for encoding")
    encode_group.add_argument(
        "--skip_encode_img", action="store_true", help="skip encoding image features, use raw HE features instead"
    )

    # Encode binsize arguments
    binsize_group = parser.add_argument_group("Encode binsize arguments")
    binsize_group.add_argument(
        "--binsize_ratio",
        type=int,
        default=2,
        help="binsize ratio for encoding. Set to 0 to only encode highres features, set to -1 to skip encoding",
    )
    binsize_group.add_argument(
        "--binsize_hidden_dim", type=int, default=128, help="hidden dimension for encoded features"
    )
    binsize_group.add_argument(
        "--compare_loss_weight", type=float, default=1.0, help="compare loss weight for encode_binsize_features"
    )
    binsize_group.add_argument("--n_pcs", type=int, default=64, help="number of PCs for encode_binsize_features")
    binsize_group.add_argument(
        "--cellbin_binsize", type=int, default=20, help="the binsize that a cellbin is similar to"
    )
    binsize_group.add_argument(
        "--tie_weights", action="store_true", help="tie weight for highres and lowres autoencoder"
    )

    # Feature combination arguments
    combine_group = parser.add_argument_group("Feature combination arguments")
    combine_group.add_argument(
        "--he_features_weight", type=float, default=1.0, help="weight for HE features during feature combination"
    )
    combine_group.add_argument(
        "--interaction_npcs", type=int, default=64, help="the pca_dim during feature interaction"
    )
    # Cluster arguments
    cluster_group = parser.add_argument_group("Cluster arguments")
    cluster_group.add_argument(
        "--n_clusters",
        type=int,
        default=10,
        help="number of clusters for kmeans. If set to <=0, don't perform kmeans clustering.",
    )
    cluster_group.add_argument(
        "--resolution",
        type=float,
        default=0,
        help="resolution for leiden. If set to <=0, don't perform leiden clustering.",
    )
    cluster_group.add_argument(
        "--included_features",
        nargs="+",
        default=["combined_with_interaction"],
        choices=[
            "encoded_he_features",
            "encoded_binsize_features",
            "binsize_he_interaction",
            "combined_standardized",
            "combined_with_interaction",
        ],
        help="list of features to include from clustering",
    )

    args = parser.parse_args()

    if args.model != "hipt" and not args.ckpt_path:
        parser.error("--ckpt_path is required when --model is not hipt (uni/uni2-h/gigapath/virchow2)")

    if args.token_size is None:
        if args.model == "hipt":
            args.token_size = 16
        else:
            args.token_size = importlib.import_module(_MODEL_MODULES[args.model]).get_token_size(
                args.model, args.ckpt_path
            )
        print(f"[token_size] auto-detected {args.token_size} for model={args.model}")

    if args.shift_margin is None:
        args.shift_margin = 16 * args.token_size
    if args.shift_stride is None:
        args.shift_stride = 4 * args.token_size

    if args.seed is not None:
        L.seed_everything(args.seed, workers=True)

    parameters = dict(sorted(vars(args).items()))

    # Create output directory
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    outfiles_name = out_dir.joinpath("outfiles.tsv")
    if not outfiles_name.exists():
        with open(outfiles_name, "w") as f:
            f.write("filename\tparameters\n")

    start_time = time.time()
    step1_start = time.time()
    # Step 1: Load and preprocess h5ad file once
    print("Step 1: Load and preprocess h5ad file once")
    adata, h5ad_path = load_and_preprocess_h5ad(
        args.h5ad,
        out_dir,
        scale=True,
        n_pcs=args.n_pcs,
        use_cache=args.use_cache,
        parameters=parameters,
        outfiles_name=outfiles_name,
    )
    step1_end = time.time()
    print(f"step 1 running time: {step1_end - step1_start} s")

    step2_start = time.time()
    # Step 2: Preprocess image
    print("Step 2: Preprocess image")
    preprocess_image_bin_size = 256
    preprocessed_img_path, bbox_path = preprocess_image(
        Path(args.input),
        adata,
        out_dir,
        bin_size=preprocess_image_bin_size,
        use_full_image=args.use_full_image,
        use_cache=args.use_cache,
        parameters=parameters,
        outfiles_name=outfiles_name,
    )
    step2_end = time.time()
    print(f"step 2 running time: {step2_end - step2_start} s")

    step3_start = time.time()
    # Step 3: Extract HE features
    print("Step 3: Extract HE features")
    image_features_st_path = extract_he_features(
        preprocessed_img_path,
        out_dir,
        device=f"cuda:{args.devices[0]}",
        shift_margin=args.shift_margin,
        shift_stride=args.shift_stride,
        model=args.model,
        token_size=args.token_size,
        img_batch_size=args.img_batch_size,
        ckpt_path=args.ckpt_path,
        use_cache=args.use_cache,
        parameters=parameters,
        outfiles_name=outfiles_name,
    )
    step3_end = time.time()
    print(f"step 3 running time: {step3_end - step3_start} s")
    gc.collect()
    torch.cuda.empty_cache()

    step4_start = time.time()
    # Step 4: Encode HE features to match binsize
    print("Step 4: Encode HE features to match binsize")
    encoded_he, encoded_he_path = encode_image_features(
        image_features_st_path,
        bbox_path=bbox_path,
        out_dir=out_dir,
        h5ad_path=h5ad_path,
        smoothen=args.smoothen,
        smoothen_size=args.smoothen_size,
        skip_encode_img=args.skip_encode_img,
        img_hidden_dim=args.img_hidden_dim,
        devices=args.devices,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        token_size=args.token_size,
        use_cache=args.use_cache,
        parameters=parameters,
        outfiles_name=outfiles_name,
    )
    step4_end = time.time()
    print(f"step 4 running time: {step4_end - step4_start} s")
    gc.collect()
    torch.cuda.empty_cache()

    step5_start = time.time()
    # Step 5: Encode binsize features
    print("Step 5: Encode binsize features")
    encoded_binsize, encoded_binsize_path = encode_binsize_features(
        h5ad_path,
        adata,
        out_dir,
        binsize_ratio=args.binsize_ratio,
        devices=args.devices,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        use_cache=args.use_cache,
        binsize_hidden_dim=args.binsize_hidden_dim,
        compare_loss_weight=args.compare_loss_weight,
        n_pcs=args.n_pcs,
        tie_weights=args.tie_weights,
        cellbin_binsize=args.cellbin_binsize,
        parameters=parameters,
        outfiles_name=outfiles_name,
    )
    step5_end = time.time()
    print(f"step 5 running time: {step5_end - step5_start} s")
    gc.collect()
    torch.cuda.empty_cache()

    step6_start = time.time()
    # Step 6: Combine features
    print("Step 6: Combine features")
    combined_features = combine_features(
        encoded_he,
        encoded_binsize,
        he_features_weight=args.he_features_weight,
        interaction_npcs=args.interaction_npcs,
    )
    step6_end = time.time()
    print(f"step 6 running time: {step6_end - step6_start} s")
    print(f"RUNING TIME: {step6_end - start_time} s")

    step7_start = time.time()
    # Step 7: Cluster features
    if args.resolution > 0 or args.n_clusters > 0:
        print("Step 7: Cluster features")
    else:
        print("WARNNING: both resolution and n_clusters are <=0, clustering step will be skipped.")
    cluster_features(
        h5ad_path,
        combined_features,
        out_dir,
        encoded_he_path,
        encoded_binsize_path,
        he_features_weight=args.he_features_weight,
        interaction_npcs=args.interaction_npcs,
        included_features=args.included_features,
        scale=True,
        n_clusters=args.n_clusters,
        resolution=args.resolution,
        use_cache=args.use_cache,
        cellbin_binsize=args.cellbin_binsize,
        parameters=parameters,
        outfiles_name=outfiles_name,
    )
    step7_end = time.time()
    print(f"step 7 running time: {step7_end - step7_start} s")
    print("Processing complete! Results saved to:", out_dir)


if __name__ == "__main__":
    main()

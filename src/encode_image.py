import os
from pathlib import Path

import anndata
import lightning as L
import pandas as pd
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.data import Dataset

from common_model import AutoEncoder
from utils_image import (
    get_bounds_from_cell_border,
    map_img_feature_to_bin,
    map_img_feature_to_cellbin,
    pool_img_feature_by_bin,
    read_img_features,
)


class ImageAE(L.LightningModule):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        lr=5e-4,
        weight_decay=1e-4,
        max_epochs=100,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        self.model = AutoEncoder(self.input_dim, self.hidden_dim)

        self.emb_result = []
        self.idx_result = []

    def forward(self, input_features):
        return self.model(input_features)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=self.lr / 50
        )
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        input_features, _ = batch
        recon_features = self.model(input_features)
        loss = self.model.loss(input_features, recon_features)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_features, idx = batch
        z = self.model.encode(input_features)
        self.emb_result.extend(z.cpu())
        self.idx_result.extend(idx.cpu())

    def on_test_epoch_end(self) -> None:
        result = sorted(
            list(zip(self.emb_result, self.idx_result)),
            key=lambda x: x[-1],  # sort by idx
        )
        encoded_features = [x[0] for x in result]
        embs = {"encoded_features": torch.stack(encoded_features)}

        if self.logger.save_dir is not None:
            save_file(embs, Path(self.logger.save_dir).joinpath("encoded_embs.safetensors"))

        self.emb_result.clear()
        self.idx_result.clear()

    def predict_step(self, batch, batch_idx):
        input_features, idx = batch
        with torch.no_grad():
            z = self.model.encode(input_features)
        return {"embeddings": z.cpu(), "indices": idx.cpu()}


class ImgAEFeatureDataset(Dataset):
    def __init__(
        self,
        h5ad,
        bbox_path,
        image_features_st,
        token_size=16,
    ):
        super().__init__()
        # read extra info
        if isinstance(h5ad, (str, os.PathLike)):
            # If h5ad is a path-like object, read and preprocess it
            adata = anndata.read_h5ad(h5ad)
        else:
            # Assume h5ad is an AnnData object
            adata = h5ad
        bbox = pd.read_csv(bbox_path, sep="\t")

        # image features
        image_features = read_img_features(image_features_st)
        if adata.uns["bin_type"] == "bins":
            img_feature_to_bin = map_img_feature_to_bin(
                adata.obsm["spatial"], adata.uns["bin_size"], bbox, token_size=token_size
            )
        elif adata.uns["bin_type"] == "cell_bins":
            cell_bounds = get_bounds_from_cell_border(adata.obsm["cell_border"], adata.obsm["spatial"])
            img_feature_to_bin = map_img_feature_to_cellbin(cell_bounds, bbox, token_size=token_size)
        else:
            raise ValueError(f"Unrecognized bin type {adata.uns['bin_type']}")

        sub_bin_pools = pool_img_feature_by_bin(img_feature_to_bin, image_features)
        self.image_features = torch.tensor(sub_bin_pools)
        self.image_features = F.normalize(self.image_features)
        self.input_dim = self.image_features.shape[-1]

    def __len__(self):
        return self.image_features.shape[0]

    def __getitem__(self, idx):
        item = self.image_features[idx]
        return item, idx

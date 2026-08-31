import os
from itertools import chain
from pathlib import Path

import anndata
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.data import Dataset

from common_model import AutoEncoder
from utils_transcriptomics import high2low, highres_adata_to_lowres, scanpy_preprocess


class BinsizeAE(L.LightningModule):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        tie_weights=False,
        compare_loss_weight=1.0,
        lr=5e-4,
        weight_decay=1e-4,
        max_epochs=100,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.compare_loss_weight = compare_loss_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        self.lowres_exp_ae = AutoEncoder(self.input_dim, self.hidden_dim)
        if tie_weights:
            self.highres_exp_ae = self.lowres_exp_ae
        else:
            self.highres_exp_ae = AutoEncoder(self.input_dim, self.hidden_dim)

        self.lowres_emb_result = []
        self.highres_emb_result = []
        self.idx_result = []
        self.highres_idx_result = []

    def forward(self, lowres_items, highres_items):
        lowres_recon = self.lowres_exp_ae(lowres_items)
        highres_recon = self.highres_exp_ae(highres_items)
        return lowres_recon, highres_recon

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=self.lr / 50
        )
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        lowres_items, highres_items_nested = batch
        lowres_z = self.lowres_exp_ae.encode(lowres_items)
        lowres_recon = self.lowres_exp_ae.decode(lowres_z)
        lowres_loss = self.lowres_exp_ae.loss(lowres_items, lowres_recon)

        highres_z = self.highres_exp_ae.encode(highres_items_nested)
        highres_recon = self.highres_exp_ae.decode(highres_z)
        highres_loss = self.highres_exp_ae.loss(
            torch.cat([a for a in highres_items_nested]), torch.cat([a for a in highres_recon])
        )  # flatten to calculate loss
        # need to pool highres_z because there are N highres bin in 1 lowres bin
        highres_z_mean = torch.zeros_like(lowres_z)
        for i, h_emb in enumerate(highres_z):
            highres_z_mean[i] = h_emb.mean(0)

        high_low_compare_loss = F.mse_loss(lowres_z, highres_z_mean)
        loss = lowres_loss + highres_loss + self.compare_loss_weight * high_low_compare_loss

        self.log("lowres_loss", lowres_loss)
        self.log("highres_loss", highres_loss)
        self.log("high_low_compare_loss", high_low_compare_loss)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        lowres_items, highres_items_nested, idx, highres_idx = batch
        lowres_z = self.lowres_exp_ae.encode(lowres_items)
        highres_items = torch.cat([a for a in highres_items_nested])  # flatten to enable reparameterize
        highres_items_nested_lens = [a.shape[0] for a in highres_items_nested]  # record lengths to split later
        highres_z = self.highres_exp_ae.encode(highres_items)
        highres_z = torch.nested.nested_tensor(list(torch.split(highres_z, highres_items_nested_lens)))

        self.lowres_emb_result.extend(lowres_z.cpu())
        self.idx_result.extend(idx.cpu())
        self.highres_emb_result.append(highres_z.cpu())
        self.highres_idx_result.append(highres_idx.cpu())

    def on_test_epoch_end(self) -> None:
        lowres_result = sorted(
            list(zip(self.lowres_emb_result, self.idx_result)),
            key=lambda x: x[1],  # sort by idx
        )
        X_lowres = [x[0] for x in lowres_result]

        highres_emb = list(chain(*chain(*self.highres_emb_result)))  # flatten
        highres_idx = list(chain(*chain(*self.highres_idx_result)))  # flatten
        highres_result = sorted(
            list(zip(highres_emb, highres_idx)),
            key=lambda x: x[1],  # sort by highres_idx
        )
        X_highres = [x[0] for x in highres_result]

        embs = {"encoded_lowres": torch.stack(X_lowres), "encoded_highres": torch.stack(X_highres)}

        if self.logger.save_dir is not None:
            save_file(embs, Path(self.logger.save_dir).joinpath("encoded_binsize_embs.safetensors"))

        self.lowres_emb_result.clear()
        self.highres_emb_result.clear()
        self.idx_result.clear()
        self.highres_idx_result.clear()

    def predict_step(self, batch, batch_idx):
        lowres_items, highres_items_nested, idx, highres_idx = batch
        lowres_z = self.lowres_exp_ae.encode(lowres_items)
        highres_items = torch.cat([a for a in highres_items_nested])  # flatten to enable reparameterize
        highres_items_nested_lens = [a.shape[0] for a in highres_items_nested]  # record lengths to split later
        highres_z = self.highres_exp_ae.encode(highres_items)
        highres_z = torch.nested.nested_tensor(list(torch.split(highres_z, highres_items_nested_lens)))

        return {
            "lowres": {"embeddings": lowres_z.cpu(), "indices": idx.cpu()},
            "highres": {"embeddings": highres_z, "indices": highres_idx.cpu()},
        }


class BinsizeAEFeatureDataset(Dataset):
    def __init__(
        self,
        highres_h5ad,
        binsize_ratio: int = 4,
        scale_h5ad=True,
        n_pcs=50,
        no_preprocess=False,
        cellbin_binsize=20,  # cellbin is similar to bin20 by default
    ):
        super().__init__()

        if isinstance(highres_h5ad, (str, os.PathLike)):
            highres_adata = anndata.read_h5ad(highres_h5ad)
            if not no_preprocess:
                highres_adata = scanpy_preprocess(
                    highres_adata, exp_mode="pca", scale=scale_h5ad, n_pcs=n_pcs, check_raw_counts=True
                )
        else:
            highres_adata = highres_h5ad
        assert binsize_ratio >= 1, "binsize_ratio must be >= 1"
        if highres_adata.uns["bin_type"] == "bins":
            highres_binsize = highres_adata.uns["bin_size"]
            lowres_binsize = highres_binsize * binsize_ratio
        elif highres_adata.uns["bin_type"] == "cell_bins":
            lowres_binsize = cellbin_binsize * binsize_ratio
        else:
            raise ValueError(f"Unrecognized bin type {highres_adata.uns['bin_type']}")
        # use raw counts for lowres conversion
        lowres_adata = highres_adata_to_lowres(highres_adata, lowres_binsize, layer="counts")
        self.lowres_spatial = lowres_adata.obsm["spatial"]
        h2l = high2low(highres_adata.obsm["spatial"], lowres_binsize)
        # Build mapping: (x, y) -> list of highres indices
        self.lowres_to_highres = {}
        for i, (x, y) in enumerate(h2l):
            key = (x, y)
            if key not in self.lowres_to_highres:
                self.lowres_to_highres[key] = []
            self.lowres_to_highres[key].append(i)

        # expression features
        lowres_adata = scanpy_preprocess(
            lowres_adata, exp_mode="pca", scale=scale_h5ad, n_pcs=n_pcs, check_raw_counts=True, verbose=True
        )
        self.lowres_item_tensors = torch.tensor(lowres_adata.obsm["X_pca"], dtype=torch.float32)
        self.highres_item_tensors = torch.tensor(highres_adata.obsm["X_pca"], dtype=torch.float32)
        self.lowres_item_tensors = F.normalize(self.lowres_item_tensors)
        self.highres_item_tensors = F.normalize(self.highres_item_tensors)
        self.input_dim = self.lowres_item_tensors.shape[-1]

    def __len__(self):
        return self.lowres_item_tensors.shape[0]

    def __getitem__(self, idx):
        lowres_pos = tuple(self.lowres_spatial[idx])
        highres_idx = self.lowres_to_highres[lowres_pos]
        lowres_item = self.lowres_item_tensors[idx]
        highres_items = self.highres_item_tensors[highres_idx]
        return lowres_item, highres_items, idx, highres_idx


def collate_highres_train(data):
    lowres_item_list, highres_items_list, _, _ = zip(*data)
    # use nested_tensor to handle items with different length (number of highres bins in 1 lowres bin)
    highres_items_nested = torch.nested.nested_tensor(list(highres_items_list))
    lowres_items = torch.vstack(lowres_item_list)
    return lowres_items, highres_items_nested


def collate_highres_test(data):
    lowres_item_list, highres_items_list, idx_list, highres_idx_list = zip(*data)
    # use nested_tensor to handle items with different length (number of highres bins in 1 lowres bin)
    highres_items_nested = torch.nested.nested_tensor(list(highres_items_list))
    lowres_items = torch.vstack(lowres_item_list)
    idx = torch.from_numpy(np.vstack(idx_list))
    highres_idx = torch.nested.nested_tensor(list(highres_idx_list))
    return lowres_items, highres_items_nested, idx, highres_idx


class HighresAE(L.LightningModule):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        tie_weights=False,  # placeholder, not used
        compare_loss_weight=None,  # placeholder, not used
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

        self.highres_exp_ae = AutoEncoder(self.input_dim, self.hidden_dim)

        self.highres_emb_result = []
        self.idx_result = []

    def forward(self, highres_items):
        highres_recon = self.highres_exp_ae(highres_items)
        return highres_recon

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=self.lr / 50
        )
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        highres_items, _ = batch
        highres_z = self.highres_exp_ae.encode(highres_items)
        highres_recon = self.highres_exp_ae.decode(highres_z)
        highres_loss = self.highres_exp_ae.loss(highres_items, highres_recon)

        loss = highres_loss
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        highres_items, idx = batch
        highres_z = self.highres_exp_ae.encode(highres_items)
        self.highres_emb_result.extend(highres_z.cpu())
        self.idx_result.extend(idx.cpu())

    def on_test_epoch_end(self) -> None:
        highres_result = sorted(
            list(zip(self.highres_emb_result, self.idx_result)),
            key=lambda x: x[1],  # sort by idx
        )
        X_highres = [x[0] for x in highres_result]

        embs = {"encoded_highres": torch.stack(X_highres)}

        if self.logger.save_dir is not None:
            save_file(embs, Path(self.logger.save_dir).joinpath("encoded_binsize_embs.safetensors"))

        self.highres_emb_result.clear()
        self.idx_result.clear()

    def predict_step(self, batch, batch_idx):
        highres_items, idx = batch
        highres_z = self.highres_exp_ae.encode(highres_items)

        return {"highres": {"embeddings": highres_z, "indices": idx.cpu()}}


class HighresAEFeatureDataset(Dataset):
    def __init__(
        self,
        highres_h5ad,
        binsize_ratio=None,  # placeholder, not used
        scale_h5ad=True,
        n_pcs=50,
        no_preprocess=False,
        cellbin_binsize=20,  # placeholder, not used
    ):
        super().__init__()

        if isinstance(highres_h5ad, (str, os.PathLike)):
            highres_adata = anndata.read_h5ad(highres_h5ad)
            if not no_preprocess:
                highres_adata = scanpy_preprocess(
                    highres_adata, exp_mode="pca", scale=scale_h5ad, n_pcs=n_pcs, check_raw_counts=True
                )
        else:
            highres_adata = highres_h5ad

        self.highres_item_tensors = torch.tensor(highres_adata.obsm["X_pca"], dtype=torch.float32)
        self.highres_item_tensors = F.normalize(self.highres_item_tensors)
        self.input_dim = self.highres_item_tensors.shape[-1]

    def __len__(self):
        return self.highres_item_tensors.shape[0]

    def __getitem__(self, idx):
        highres_items = self.highres_item_tensors[idx]
        return highres_items, idx


def collate_highres_only(data):
    highres_items_list, idx_list = zip(*data)
    # use nested_tensor to handle items with different length (number of highres bins in 1 lowres bin)
    highres_items = torch.vstack(highres_items_list)
    idx = torch.from_numpy(np.vstack(idx_list))
    return highres_items, idx

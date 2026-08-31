import torch.nn as nn


def make_mlp(
    layer_sizes,
    activation=nn.GELU,
    batchnorm=False,  # NOTE: nested tensor does not support batch norm
):
    layers = []
    for i in range(len(layer_sizes) - 1):
        in_ch = layer_sizes[i]
        out_ch = layer_sizes[i + 1]
        layers.append(nn.Linear(in_ch, out_ch))
        if batchnorm:
            layers.append(nn.BatchNorm1d(out_ch))
        if i < len(layer_sizes) - 2:  # not last layer
            layers.append(activation())
    return nn.Sequential(*layers)


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = make_mlp([input_dim, latent_dim * 2, latent_dim])
        self.decoder = make_mlp([latent_dim, latent_dim * 2, input_dim])

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon

    def recon_loss(self, x, x_recon):
        recon = nn.MSELoss()
        return recon(x_recon, x)

    def loss(self, x, x_recon):
        return self.recon_loss(x, x_recon)

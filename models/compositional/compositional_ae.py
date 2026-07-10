import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from models.compositional.networks import FieldEncoder, FieldDecoder, RegimeHead, SDFHead


def cross_block_correlation(a, b, eps=1e-8):
    """Mean absolute Pearson correlation between coordinates of two latent blocks
    (loss L6 of the working notes), computed across the batch."""
    if a.shape[0] < 2:
        return a.new_zeros(())
    a = (a - a.mean(0)) / (a.std(0) + eps)
    b = (b - b.mean(0)) / (b.std(0) + eps)
    corr = a.t() @ b / (a.shape[0] - 1)
    return corr.abs().mean()


class CompositionalAE(pl.LightningModule):
    """
    Path A of the compositional framework (working notes §12), specialized to
    steady lid-driven cavity flow with a geometry inside.

    Block-structured latent z = [z_mu || z_g || z_xi]:
        z_mu : regime block, aligned with log Re via a regression head
        z_g  : geometry block, aligned with the object SDF via a small SDF decoder
        z_xi : residual block, absorbs whatever reconstruction needs beyond (mu, g)

    The flow is steady, so there is no dynamics block z_eta and no propagator.

    Losses: masked L2 reconstruction (L1), regime supervision, geometry
    supervision, and Pearson cross-block decorrelation (L6).
    """

    def __init__(self, in_channels=3, resolution=256, base_channels=32,
                 latent_mu=4, latent_g=32, latent_xi=16, sdf_resolution=64,
                 lambda_recon=1.0, lambda_regime=0.1, lambda_geo=0.1,
                 lambda_decorr=0.01, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        latent_dim = latent_mu + latent_g + latent_xi
        self.encoder = FieldEncoder(in_channels, resolution, base_channels,
                                    latent_mu, latent_g, latent_xi)
        self.decoder = FieldDecoder(latent_dim, in_channels, resolution, base_channels)
        self.regime_head = RegimeHead(latent_mu)
        self.sdf_head = SDFHead(latent_g, sdf_resolution, base_channels)

    def encode(self, fields):
        z_mu, z_g, z_xi = self.encoder(fields)
        return z_mu, z_g, z_xi

    def forward(self, fields):
        z_mu, z_g, z_xi = self.encode(fields)
        recon = self.decoder(torch.cat([z_mu, z_g, z_xi], dim=1))
        return recon, (z_mu, z_g, z_xi)

    def masked_recon_loss(self, recon, fields, mask, per_channel=False):
        """MSE over the fluid region only, normalized by fluid node count."""
        recon = recon * mask  # zero out predictions inside the solid
        se = (recon - fields) ** 2
        node_count = mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
        full = (se.sum(dim=(1, 2, 3)) / node_count).mean()
        if not per_channel:
            return full
        per = [(se[:, i].sum(dim=(1, 2)) / node_count).mean() for i in range(se.shape[1])]
        return full, per

    def _losses(self, batch):
        fields, mask = batch['fields'], batch['mask']
        recon, (z_mu, z_g, z_xi) = self(fields)

        loss_recon = self.masked_recon_loss(recon, fields, mask)
        loss_regime = F.mse_loss(self.regime_head(z_mu), batch['log_re'])

        sdf_lr = F.interpolate(batch['sdf'], size=self.sdf_head.resolution,
                               mode='bilinear', align_corners=False)
        loss_geo = F.mse_loss(self.sdf_head(z_g), sdf_lr)

        loss_decorr = (cross_block_correlation(z_mu, z_g)
                       + cross_block_correlation(z_mu, z_xi)
                       + cross_block_correlation(z_g, z_xi)) / 3.0

        h = self.hparams
        total = (h.lambda_recon * loss_recon + h.lambda_regime * loss_regime
                 + h.lambda_geo * loss_geo + h.lambda_decorr * loss_decorr)
        return total, {'recon': loss_recon, 'regime': loss_regime,
                       'geo': loss_geo, 'decorr': loss_decorr}

    def training_step(self, batch, batch_idx):
        total, parts = self._losses(batch)
        self.log('train_loss', total, prog_bar=True)
        for name, value in parts.items():
            self.log(f'train_loss_{name}', value)
        return total

    def validation_step(self, batch, batch_idx):
        total, parts = self._losses(batch)
        self.log('val_loss', total, prog_bar=True)
        for name, value in parts.items():
            self.log(f'val_loss_{name}', value)

        # per-channel reconstruction errors, as in the FlowBench template
        recon, _ = self(batch['fields'])
        full, per = self.masked_recon_loss(recon, batch['fields'], batch['mask'],
                                           per_channel=True)
        self.log('val_loss_full', full, on_epoch=True)
        for name, value in zip(['u', 'v', 'p'], per):
            self.log(f'val_loss_{name}', value, on_epoch=True)
        return total

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

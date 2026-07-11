import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from models.compositional.networks import FieldEncoder, FieldDecoder, RegimeHead, SDFHead


def same_factor_invariance(z, group_ids):
    """Same-factor invariance (loss L10 of the working notes): penalize the
    variance of a latent block across batch samples that share a factor
    (here: the same geometry at different Re). Requires group-structured
    minibatches so that groups actually co-occur in the batch."""
    loss = z.new_zeros(())
    count = 0
    for gid in group_ids.unique():
        sel = group_ids == gid
        if sel.sum() >= 2:
            zg = z[sel]
            loss = loss + ((zg - zg.mean(0, keepdim=True)) ** 2).mean()
            count += 1
    return loss / count if count else loss


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
                 lambda_decorr=0.01, lambda_inv=0.0, lambda_swap=0.0,
                 lambda_xswap=0.0, lr=1e-3):
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

    def swap_consistency(self, z_mu, z_g, z_xi, batch):
        """Swap consistency (loss L12 of the working notes): for in-batch pairs
        (i, k) at the same Re but different geometry, decoding [z_mu from i ||
        z_g, z_xi from k] must reproduce sample k's true field. This demands the
        regime code be functionally interchangeable across geometries at fixed
        dynamic similarity. Returns (loss, number_of_pairs)."""
        re, gid = batch['re'], batch['geo_id']
        n = re.shape[0]
        same_re = re.view(-1, 1) == re.view(1, -1)
        diff_geo = gid.view(-1, 1) != gid.view(1, -1)
        idx_i, idx_k = torch.nonzero(same_re & diff_geo, as_tuple=True)
        if idx_i.numel() == 0:
            return z_mu.new_zeros(()), 0
        if idx_i.numel() > n:  # cap decode cost at one extra batch
            keep = torch.randperm(idx_i.numel(), device=idx_i.device)[:n]
            idx_i, idx_k = idx_i[keep], idx_k[keep]
        z_swap = torch.cat([z_mu[idx_i], z_g[idx_k], z_xi[idx_k]], dim=1)
        recon = self.decoder(z_swap)
        loss = self.masked_recon_loss(recon, batch['fields'][idx_k],
                                      batch['mask'][idx_k])
        return loss, idx_i.numel()

    def cross_swap_consistency(self, z_mu, z_g, z_xi, batch):
        """Cross-Re swap (training form of the transfer test): for in-batch
        triples (i, k, m) with re_m == re_i, geo_m == geo_k != geo_i and
        re_k != re_i, decoding [z_mu from i || z_g, z_xi from k] must
        reproduce sample m — geometry k moved to operating point Re_i. This
        forces the decoder to take Re from z_mu and treat (z_g, z_xi) as
        Re-free. Returns (loss, number_of_triples)."""
        re, gid = batch['re'], batch['geo_id']
        n = re.shape[0]
        same_re = re.view(-1, 1) == re.view(1, -1)
        diff_geo = gid.view(-1, 1) != gid.view(1, -1)
        pair_i, pair_m = torch.nonzero(same_re & diff_geo, as_tuple=True)
        idx_i, idx_k, idx_m = [], [], []
        for i, m in zip(pair_i.tolist(), pair_m.tolist()):
            cand = torch.nonzero((gid == gid[m]) & (re != re[i])).flatten()
            if cand.numel():
                k = cand[torch.randint(cand.numel(), (1,))].item()
                idx_i.append(i); idx_k.append(k); idx_m.append(m)
        if not idx_i:
            return z_mu.new_zeros(()), 0
        if len(idx_i) > n:  # cap decode cost at one extra batch
            keep = torch.randperm(len(idx_i))[:n].tolist()
            idx_i = [idx_i[j] for j in keep]
            idx_k = [idx_k[j] for j in keep]
            idx_m = [idx_m[j] for j in keep]
        z_swap = torch.cat([z_mu[idx_i], z_g[idx_k], z_xi[idx_k]], dim=1)
        recon = self.decoder(z_swap)
        loss = self.masked_recon_loss(recon, batch['fields'][idx_m],
                                      batch['mask'][idx_m])
        return loss, len(idx_i)

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

        # L10: z_g must not change across samples sharing a geometry
        loss_inv = same_factor_invariance(z_g, batch['geo_id'])

        # L12: z_mu must be recombinable across geometries at the same Re
        h = self.hparams
        if h.lambda_swap > 0:
            loss_swap, n_pairs = self.swap_consistency(z_mu, z_g, z_xi, batch)
        else:
            loss_swap, n_pairs = loss_recon.new_zeros(()), 0

        # cross-Re swap: z_mu must transfer a geometry to a new operating point
        if h.lambda_xswap > 0:
            loss_xswap, n_triples = self.cross_swap_consistency(z_mu, z_g, z_xi, batch)
        else:
            loss_xswap, n_triples = loss_recon.new_zeros(()), 0

        total = (h.lambda_recon * loss_recon + h.lambda_regime * loss_regime
                 + h.lambda_geo * loss_geo + h.lambda_decorr * loss_decorr
                 + h.lambda_inv * loss_inv + h.lambda_swap * loss_swap
                 + h.lambda_xswap * loss_xswap)
        return total, {'recon': loss_recon, 'regime': loss_regime,
                       'geo': loss_geo, 'decorr': loss_decorr, 'inv': loss_inv,
                       'swap': loss_swap, 'swap_pairs': float(n_pairs),
                       'xswap': loss_xswap, 'xswap_triples': float(n_triples)}

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

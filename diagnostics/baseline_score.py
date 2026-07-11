"""
FlowBench baseline comparison (project-plan success criterion 1).

Computes the benchmark paper's M1/M2 scores (Rabeh et al., Communications
Engineering 4:182, 2025) for our model, evaluated at the benchmark's native
512x512 resolution:

    M1: per-pixel MSE over the fluid region (channels summed, divided by the
        fluid-pixel count), averaged over the test set.
    M2: the same, restricted to the boundary layer 0 <= SDF <= 0.2.
    score = -(100/6) * log10(MSE)   (100 <-> MSE 1e-6, 0 <-> MSE 1)

Two evaluation modes:
    reconstruction — encode the target field itself (an upper bound; the
        baselines never see the answer).
    donor prediction — the operator-comparable mode for static-geometry
        models: z_g from the TARGET's SDF, z_mu and z_xi from a donor flow at
        the same Re but a different geometry. The model never sees the
        target's field.

Usage:
    python diagnostics/baseline_score.py --config configs/compositional/run7.yaml \
        --checkpoint checkpoints/compositional-run7/version_0/cae-epoch=188.ckpt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from data.dataset import CompositionalLDCDataset
from models.compositional.compositional_ae import CompositionalAE


def score(mse):
    return -(100.0 / 6.0) * np.log10(max(mse, 1e-300))


def masked_mse(pred, truth, region):
    """Per-sample MSE with channels summed and divided by region pixel count,
    matching the benchmark's convention. pred/truth: (B,3,H,W); region:
    (B,1,H,W) binary."""
    se = ((pred - truth) ** 2 * region).sum(dim=(1, 2, 3))
    count = region.sum(dim=(1, 2, 3)).clamp(min=1.0)
    return se / count


def find_same_re_donors(re, gid):
    """For each sample, the index of a same-Re different-geometry donor, or -1."""
    donors = torch.full((len(re),), -1, dtype=torch.long)
    by_re = {}
    for i, r in enumerate(re.tolist()):
        by_re.setdefault(r, []).append(i)
    for i, (r, g) in enumerate(zip(re.tolist(), gid.tolist())):
        for j in by_re[r]:
            if gid[j] != g:
                donors[i] = j
                break
    return donors


@torch.no_grad()
def main(config_path, checkpoint_path, batch_size=8):
    config = OmegaConf.load(config_path)

    # low-resolution dataset for the encoders (training resolution)
    train_dataset = CompositionalLDCDataset(
        file_path_x=config.data.file_path_train_x,
        file_path_y=config.data.file_path_train_y,
        resolution=config.data.resolution,
    )
    dataset = CompositionalLDCDataset(
        file_path_x=config.data.file_path_test_x,
        file_path_y=config.data.file_path_test_y,
        resolution=config.data.resolution,
        re_stats=train_dataset.re_stats,
    )

    # full-resolution ground truth for scoring at the benchmark's 512^2
    x_raw = np.load(config.data.file_path_test_x)['data']
    y_raw = np.load(config.data.file_path_test_y)['data']
    truth512 = torch.tensor(y_raw[:, :3], dtype=torch.float32)
    sdf512 = torch.tensor(x_raw[:, 1:2], dtype=torch.float32)
    mask512 = torch.tensor(x_raw[:, 2:3], dtype=torch.float32)
    if mask512.max() > 1.0:
        mask512 = mask512 / 255.0
    fluid512 = (mask512 > 0.5).float()
    band512 = fluid512 * ((sdf512 >= 0) & (sdf512 <= 0.2)).float()
    hi_res = truth512.shape[-1]

    model = CompositionalAE.load_from_checkpoint(checkpoint_path, map_location='cpu')
    model.eval()

    donors = find_same_re_donors(dataset.re, dataset.geo_ids)
    n_donor = int((donors >= 0).sum())
    print(f'{n_donor}/{len(dataset)} test samples have a same-Re donor '
          f'(donor-prediction mode covers these).')

    metrics = {'recon': {'m1': [], 'm2': []}, 'donor': {'m1': [], 'm2': []}}
    for b in range(0, len(dataset), batch_size):
        idx = torch.arange(b, min(b + batch_size, len(dataset)))
        fields = dataset.fields[idx]
        sdf = dataset.sdf[idx]

        # mode 1: reconstruction (sees the target field)
        z_mu, z_g, z_xi = model.encode(fields, sdf)
        recon = model.decoder(torch.cat([z_mu, z_g, z_xi], dim=1))
        recon = F.interpolate(recon, size=(hi_res, hi_res), mode='bilinear',
                              align_corners=False)
        metrics['recon']['m1'] += masked_mse(recon, truth512[idx], fluid512[idx]).tolist()
        metrics['recon']['m2'] += masked_mse(recon, truth512[idx], band512[idx]).tolist()

        # mode 2: donor prediction (target SDF + same-Re donor flow only)
        has_donor = donors[idx] >= 0
        if has_donor.any():
            tgt = idx[has_donor]
            dnr = donors[tgt]
            z_mu_d, _, z_xi_d = model.encode(dataset.fields[dnr], dataset.sdf[dnr])
            if model.hparams.get('static_geometry', False):
                z_g_t = model.geom_encoder(dataset.sdf[tgt])
            else:
                _, z_g_t, _ = model.encode(dataset.fields[tgt], dataset.sdf[tgt])
            pred = model.decoder(torch.cat([z_mu_d, z_g_t, z_xi_d], dim=1))
            pred = F.interpolate(pred, size=(hi_res, hi_res), mode='bilinear',
                                 align_corners=False)
            metrics['donor']['m1'] += masked_mse(pred, truth512[tgt], fluid512[tgt]).tolist()
            metrics['donor']['m2'] += masked_mse(pred, truth512[tgt], band512[tgt]).tolist()

    print(f'\nScores at {hi_res}x{hi_res} (benchmark convention: channels '
          f'summed / fluid pixels; score = -(100/6) log10 MSE):\n')
    print(f'{"mode":<22}{"M1 MSE":>12}{"M1 score":>10}{"M2 MSE":>12}{"M2 score":>10}')
    print('-' * 66)
    for mode, label in [('recon', 'reconstruction'), ('donor', 'donor prediction')]:
        if not metrics[mode]['m1']:
            continue
        m1 = float(np.mean(metrics[mode]['m1']))
        m2 = float(np.mean(metrics[mode]['m2']))
        print(f'{label:<22}{m1:>12.3e}{score(m1):>10.1f}{m2:>12.3e}{score(m2):>10.1f}')
    print('\nPublished baselines (SDF, random split, Table 1 of the benchmark '
          'paper): poseidon-T M1=64.9 M2=73.3; scOT-T 64.6/71.4; '
          'geometric-deeponet 53.0/59.9; DeepONet 45.9/53.0; CNO 44.8/54.5; '
          'FNO 44.3/59.2; WNO 24.1/41.3.')

    out_dir = os.path.dirname(checkpoint_path) or '.'
    out_path = os.path.join(out_dir, 'baseline_score_test.csv')
    with open(out_path, 'w') as f:
        f.write('mode,m1_mse,m1_score,m2_mse,m2_score\n')
        for mode in ('recon', 'donor'):
            if not metrics[mode]['m1']:
                continue
            m1 = float(np.mean(metrics[mode]['m1']))
            m2 = float(np.mean(metrics[mode]['m2']))
            f.write(f'{mode},{m1:.6e},{score(m1):.2f},{m2:.6e},{score(m2):.2f}\n')
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FlowBench M1/M2 baseline scores.')
    parser.add_argument('--config', type=str, default='configs/compositional/run7.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    main(args.config, args.checkpoint)

"""
Linear-probe diagnostic (working notes §4 / project plan Phase 1).

Fits cross-validated ridge regressions from each latent block to known
physical factors and reports R^2. For a compositional latent we expect:

    log Re   : high R^2 from z_mu, low from z_g
    geometry : high R^2 from z_g,  low from z_mu

Usage:
    python diagnostics/probes.py --config configs/compositional/conf.yaml \
        --checkpoint checkpoints/compositional/<run>/last.ckpt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from torch.utils import data

from data.dataset import CompositionalLDCDataset
from models.compositional.compositional_ae import CompositionalAE


@torch.no_grad()
def encode_dataset(model, dataset, batch_size=16, device='cpu'):
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    blocks = {'z_mu': [], 'z_g': [], 'z_xi': []}
    for batch in loader:
        z_mu, z_g, z_xi = model.encode(batch['fields'].to(device))
        blocks['z_mu'].append(z_mu.cpu())
        blocks['z_g'].append(z_g.cpu())
        blocks['z_xi'].append(z_xi.cpu())
    return {k: torch.cat(v).numpy() for k, v in blocks.items()}


def probe_r2(z_block, target, n_folds=5, alpha=1.0):
    """5-fold cross-validated R^2 of a ridge probe target ~ W z_block + b."""
    return cross_val_score(Ridge(alpha=alpha), z_block, target,
                           cv=n_folds, scoring='r2').mean()


@torch.no_grad()
def swap_error(model, dataset, max_pairs=256, batch_size=16, seed=0):
    """
    Swap-consistency error (the metric L12 trains): for test pairs (i, k) at
    the same Re but different geometry, decode [z_mu(i) || z_g(k) || z_xi(k)]
    and compare against sample k's true field. Reported alongside the ordinary
    reconstruction error on the same samples; a ratio near 1 means the blocks
    are functionally recombinable through the decoder.

    Returns (mean_swap_error, mean_recon_error, n_pairs), or None if the
    dataset contains no same-Re/different-geometry pairs.
    """
    re, gid = dataset.re, dataset.geo_ids
    same_re = re.view(-1, 1) == re.view(1, -1)
    diff_geo = gid.view(-1, 1) != gid.view(1, -1)
    idx_i, idx_k = torch.nonzero(same_re & diff_geo, as_tuple=True)
    if idx_i.numel() == 0:
        return None
    gen = torch.Generator().manual_seed(seed)
    keep = torch.randperm(idx_i.numel(), generator=gen)[:max_pairs]
    idx_i, idx_k = idx_i[keep], idx_k[keep]

    return _paired_decode_error(model, dataset, idx_i, idx_k, idx_k, batch_size)


@torch.no_grad()
def cross_re_swap_error(model, dataset, max_pairs=256, batch_size=16, seed=0):
    """
    Cross-Re swap (the demanding transfer test): take z_mu from sample i at
    Re_i and z_g, z_xi from sample k of a DIFFERENT geometry at a DIFFERENT
    Re, decode, and compare against the true field of geometry k at Re_i,
    which the dataset contains as sample m. Unlike the same-Re swap, this
    cannot be passed by z_mu being a deterministic function of Re: the decoder
    must genuinely compose the regime code with a geometry it never saw at
    that operating point in this latent combination.
    """
    re, gid = dataset.re, dataset.geo_ids
    lookup = {(r, g): m for m, (r, g) in enumerate(zip(re.tolist(), gid.tolist()))}
    idx_i, idx_k, idx_m = [], [], []
    for i, (r_i, g_i) in enumerate(zip(re.tolist(), gid.tolist())):
        for k, (r_k, g_k) in enumerate(zip(re.tolist(), gid.tolist())):
            if g_i != g_k and r_i != r_k:
                m = lookup.get((r_i, g_k))
                if m is not None:
                    idx_i.append(i); idx_k.append(k); idx_m.append(m)
    if not idx_i:
        return None
    gen = torch.Generator().manual_seed(seed)
    keep = torch.randperm(len(idx_i), generator=gen)[:max_pairs]
    to_t = lambda lst: torch.tensor(lst)[keep]
    return _paired_decode_error(model, dataset, to_t(idx_i), to_t(idx_k),
                                to_t(idx_m), batch_size)


@torch.no_grad()
def _paired_decode_error(model, dataset, idx_i, idx_k, idx_m, batch_size):
    """Decode [z_mu(i) || z_g(k) || z_xi(k)] and compare against sample m's
    truth; also compute m's ordinary reconstruction error, and the error
    against the donor k's field (if the swapped output matches k rather than
    m, the decoder is reading Re from z_g/z_xi instead of z_mu)."""
    swap_sum, recon_sum, donor_sum, n = 0.0, 0.0, 0.0, 0
    for b in range(0, idx_i.numel(), batch_size):
        bi, bk, bm = idx_i[b:b + batch_size], idx_k[b:b + batch_size], idx_m[b:b + batch_size]
        z_mu_i, _, _ = model.encode(dataset.fields[bi])
        _, z_g_k, z_xi_k = model.encode(dataset.fields[bk])
        z_mu_m, z_g_m, z_xi_m = model.encode(dataset.fields[bm])
        fields_m, mask_m = dataset.fields[bm], dataset.mask[bm]

        recon_swap = model.decoder(torch.cat([z_mu_i, z_g_k, z_xi_k], dim=1))
        recon_self = model.decoder(torch.cat([z_mu_m, z_g_m, z_xi_m], dim=1))

        cnt = len(bi)
        swap_sum += model.masked_recon_loss(recon_swap, fields_m, mask_m).item() * cnt
        recon_sum += model.masked_recon_loss(recon_self, fields_m, mask_m).item() * cnt
        donor_sum += model.masked_recon_loss(recon_swap, dataset.fields[bk],
                                             dataset.mask[bk]).item() * cnt
        n += cnt
    return swap_sum / n, recon_sum / n, donor_sum / n, n


def main(config_path, checkpoint_path, split='test'):
    config = OmegaConf.load(config_path)

    train_dataset = CompositionalLDCDataset(
        file_path_x=config.data.file_path_train_x,
        file_path_y=config.data.file_path_train_y,
        resolution=config.data.resolution,
    )
    if split == 'test':
        dataset = CompositionalLDCDataset(
            file_path_x=config.data.file_path_test_x,
            file_path_y=config.data.file_path_test_y,
            resolution=config.data.resolution,
            re_stats=train_dataset.re_stats,
        )
    else:
        dataset = train_dataset

    model = CompositionalAE.load_from_checkpoint(checkpoint_path, map_location='cpu')
    model.eval()

    blocks = encode_dataset(model, dataset)
    geo = dataset.geometry_descriptors().numpy()
    targets = {
        'log_re': dataset.log_re.numpy(),
        'area_frac': geo[:, 0],
        'centroid_x': geo[:, 1],
        'centroid_y': geo[:, 2],
    }

    rows = []
    print(f'\nLinear-probe R^2 on the {split} split '
          f'({len(dataset)} samples)\n')
    header = f'{"target":<12}' + ''.join(f'{b:>10}' for b in blocks)
    print(header)
    print('-' * len(header))
    for tname, tval in targets.items():
        r2s = [probe_r2(blocks[b], tval) for b in blocks]
        rows.append([tname] + r2s)
        print(f'{tname:<12}' + ''.join(f'{r2:>10.3f}' for r2 in r2s))

    print('\nExpected pattern: log_re high from z_mu only; '
          'geometry targets high from z_g only.')

    for name, fn in [('Same-Re swap', swap_error), ('Cross-Re swap', cross_re_swap_error)]:
        result = fn(model, dataset)
        if result is None:
            print(f'\n{name}: no eligible pairs in this split.')
            continue
        swap_mse, recon_mse, donor_mse, n_pairs = result
        print(f'\n{name} error ({n_pairs} pairs): '
              f'swap={swap_mse:.3e}  recon={recon_mse:.3e}  '
              f'ratio={swap_mse / max(recon_mse, 1e-12):.2f}')
        if name == 'Cross-Re swap':
            print(f'  swapped output vs DONOR field (geometry k at its own Re): '
                  f'{donor_mse:.3e}')
            print('  (donor error << target error means the decoder reads Re '
                  'from z_g/z_xi, ignoring z_mu)')
    print('(ratio near 1 = blocks recombine cleanly through the decoder; '
          'cross-Re is the demanding transfer test)')

    out_dir = os.path.dirname(checkpoint_path) or '.'
    out_path = os.path.join(out_dir, f'probe_r2_{split}.csv')
    with open(out_path, 'w') as f:
        f.write('target,' + ','.join(blocks) + '\n')
        for row in rows:
            f.write(row[0] + ',' + ','.join(f'{v:.6f}' for v in row[1:]) + '\n')
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear-probe diagnostics.')
    parser.add_argument('--config', type=str, default='configs/compositional/conf.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'])
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.split)

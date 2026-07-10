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

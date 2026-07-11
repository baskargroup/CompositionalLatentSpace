"""
Concept-vector arithmetic diagnostic (anchor paper 4; AeroJEPA-style).

For each known parameter (log Re, area fraction, centroid x/y) we fit a
ridge probe on the full latent and take its weight direction as that
parameter's "concept vector". We then walk test latents one standardized
unit along a parameter's direction, DECODE the field, RE-ENCODE it, and
read out all parameters with the same probes. The resulting sensitivity
matrix S[k, j] = (change in parameter j when walking parameter k's
direction), in standard-deviation units.

For a compositional latent: diagonal entries near 1 (the intended
parameter responds by the intended amount) and off-diagonal entries near
0 (no cross-talk). The project plan's success criterion: diagonal >= 0.9,
off-diagonal <= 0.3.

Usage:
    python diagnostics/concept_vectors.py --config configs/compositional/conf.yaml \
        --checkpoint checkpoints/compositional/version_6/cae-epoch=190.ckpt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.linear_model import Ridge

from data.dataset import CompositionalLDCDataset
from diagnostics.probes import encode_dataset
from models.compositional.compositional_ae import CompositionalAE


@torch.no_grad()
def decode_reencode(model, z, sdf, batch_size=16):
    """Decode latents to fields, re-encode the fields, return new latents.
    `sdf` is the base samples' SDF, needed by static-geometry models (note:
    for such models geometry walks are pinned by the supplied SDF, so this
    diagnostic is most meaningful for flow-encoded geometry models)."""
    out = []
    for b in range(0, z.shape[0], batch_size):
        fields = model.decoder(z[b:b + batch_size])
        z_mu, z_g, z_xi = model.encode(fields, sdf[b:b + batch_size])
        out.append(torch.cat([z_mu, z_g, z_xi], dim=1))
    return torch.cat(out)


def main(config_path, checkpoint_path, n_samples=64, step=1.0, seed=0,
         restrict_blocks=False):
    config = OmegaConf.load(config_path)

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

    model = CompositionalAE.load_from_checkpoint(checkpoint_path, map_location='cpu')
    model.eval()

    # 1. Encode the test set and assemble the full latent matrix
    blocks = encode_dataset(model, dataset)
    Z = np.concatenate([blocks['z_mu'], blocks['z_g'], blocks['z_xi']], axis=1)
    n_mu = blocks['z_mu'].shape[1]
    n_g = blocks['z_g'].shape[1]
    # column mask per parameter when --restrict-blocks is set: the regime
    # probe may only use z_mu, the geometry probes only z_g
    block_cols = {
        'log_re': np.arange(0, n_mu),
        'area_frac': np.arange(n_mu, n_mu + n_g),
        'centroid_x': np.arange(n_mu, n_mu + n_g),
        'centroid_y': np.arange(n_mu, n_mu + n_g),
    }

    # 2. Standardized parameter targets (so responses are in sigma units)
    geo = dataset.geometry_descriptors().numpy()
    raw = {
        'log_re': dataset.log_re.numpy(),
        'area_frac': geo[:, 0],
        'centroid_x': geo[:, 1],
        'centroid_y': geo[:, 2],
    }
    names = list(raw.keys())
    stats = {k: (v.mean(), v.std() + 1e-12) for k, v in raw.items()}
    targets = {k: (v - stats[k][0]) / stats[k][1] for k, v in raw.items()}

    # 3. One ridge probe per parameter; the weight direction is the
    #    parameter's concept vector. With --restrict-blocks, each probe only
    #    sees its own block, so a walk cannot touch the wrong block by
    #    construction (its coefficients are zero elsewhere).
    probes, coefs = {}, {}
    for k, y in targets.items():
        cols = block_cols[k] if restrict_blocks else np.arange(Z.shape[1])
        p = Ridge(alpha=1.0).fit(Z[:, cols], y)
        w = np.zeros(Z.shape[1])
        w[cols] = p.coef_
        probes[k] = (p, cols)
        coefs[k] = w
    directions = {k: w / np.linalg.norm(w) for k, w in coefs.items()}

    def predict(k, z_matrix):
        p, cols = probes[k]
        return p.predict(z_matrix[:, cols])

    # 4. Walk a subset of test latents along each concept vector, decode,
    #    re-encode, and read out every parameter
    rng = np.random.default_rng(seed)
    idx = rng.choice(Z.shape[0], size=min(n_samples, Z.shape[0]), replace=False)
    z_base = torch.tensor(Z[idx], dtype=torch.float32)
    sdf_base = dataset.sdf[torch.tensor(idx)]

    # baseline readout after one decode/re-encode pass (cancels autoencoding bias)
    z_cycle = decode_reencode(model, z_base, sdf_base).numpy()
    base_pred = {k: predict(k, z_cycle) for k in names}

    S = np.zeros((len(names), len(names)))
    for a, k in enumerate(names):
        # step size chosen so the probe's own prediction moves by `step` sigma
        alpha = step / np.linalg.norm(coefs[k])
        z_walk = z_base + torch.tensor(directions[k], dtype=torch.float32) * alpha
        z_walk_cycle = decode_reencode(model, z_walk, sdf_base).numpy()
        for b, j in enumerate(names):
            S[a, b] = (predict(j, z_walk_cycle) - base_pred[j]).mean() / step

    mode = 'block-restricted probes' if restrict_blocks else 'full-latent probes'
    print(f'\nConcept-vector sensitivity matrix ({len(idx)} base samples, '
          f'step = {step} sigma, {mode}, decode -> re-encode -> probe):')
    print('rows = direction walked, columns = parameter that responded\n')
    row_label = 'walk \\ read'
    header = f'{row_label:<14}' + ''.join(f'{j:>12}' for j in names)
    print(header)
    print('-' * len(header))
    for a, k in enumerate(names):
        print(f'{k:<14}' + ''.join(f'{S[a, b]:>12.3f}' for b in range(len(names))))
    print('\nTarget: diagonal >= 0.9 (intended response), '
          'off-diagonal <= 0.3 (no cross-talk).')

    out_dir = os.path.dirname(checkpoint_path) or '.'
    suffix = '_blockwise' if restrict_blocks else ''
    out_path = os.path.join(out_dir, f'concept_matrix{suffix}_test.csv')
    with open(out_path, 'w') as f:
        f.write('walk,' + ','.join(names) + '\n')
        for a, k in enumerate(names):
            f.write(k + ',' + ','.join(f'{S[a, b]:.6f}' for b in range(len(names))) + '\n')
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Concept-vector arithmetic diagnostic.')
    parser.add_argument('--config', type=str, default='configs/compositional/conf.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--n-samples', type=int, default=64)
    parser.add_argument('--step', type=float, default=1.0)
    parser.add_argument('--restrict-blocks', action='store_true',
                        help='fit each probe only on its own block, so walks '
                             'cannot touch the wrong block by construction')
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.n_samples, args.step,
         restrict_blocks=args.restrict_blocks)

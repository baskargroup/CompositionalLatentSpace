"""
Paper figures from a trained checkpoint.

Figure A (reconstruction): for a few test samples, a 3x3 panel per sample ---
rows u, v, p; columns CFD truth / model prediction / |error|.

Figure B (cross-Re transfer): the headline result made visible. Take z_mu from
a sample at Re_i and z_g, z_xi from a different geometry observed at a
different Re_k; decode; compare against the CFD truth of that geometry at
Re_i. Columns: donor truth (geo at Re_k) / model prediction (geo at Re_i) /
CFD truth (geo at Re_i) / |error|. The triple with the largest Re jump is
chosen automatically.

Usage (on the machine with data + checkpoint):
    python plotting/figures.py --config configs/compositional/run7.yaml \
        --checkpoint checkpoints/compositional-run7/version_0/cae-epoch=188.ckpt
Outputs PNG + PDF into docs/figures/.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from data.dataset import CompositionalLDCDataset
from models.compositional.compositional_ae import CompositionalAE

CHANNELS = ['u', 'v', 'p']
FIELD_CMAP = plt.get_cmap('RdBu_r').copy()   # diverging: two hues + neutral midpoint
ERROR_CMAP = plt.get_cmap('magma').copy()    # sequential: one hue, light -> dark
FIELD_CMAP.set_bad('0.35')                   # solid object rendered dark gray
ERROR_CMAP.set_bad('0.35')


def masked(field, mask):
    out = np.array(field, dtype=float)
    out[mask < 0.5] = np.nan
    return out


def _panel_row(axs, row, images, mask, vmax, err_vmax):
    """Render one channel row: n-1 field panels sharing a symmetric scale,
    plus a final |error| panel. Returns the two image handles for colorbars."""
    im_field = None
    for col, img in enumerate(images[:-1]):
        im_field = axs[row, col].imshow(masked(img, mask), cmap=FIELD_CMAP,
                                        vmin=-vmax, vmax=vmax, origin='lower')
    im_err = axs[row, len(images) - 1].imshow(masked(images[-1], mask),
                                              cmap=ERROR_CMAP, vmin=0,
                                              vmax=err_vmax, origin='lower')
    for col in range(len(images)):
        axs[row, col].set_xticks([])
        axs[row, col].set_yticks([])
    return im_field, im_err


def _save(fig, outdir, name):
    for ext in ('png', 'pdf'):
        path = os.path.join(outdir, f'{name}.{ext}')
        fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {os.path.join(outdir, name)}.png / .pdf')


@torch.no_grad()
def reconstruction_figure(model, dataset, idx, outdir):
    sample = dataset[idx]
    fields = sample['fields'].unsqueeze(0)
    recon, _ = model(fields, sample['sdf'].unsqueeze(0))
    truth = fields[0].numpy()
    pred = recon[0].numpy()
    mask = sample['mask'][0].numpy()

    fig, axs = plt.subplots(3, 3, figsize=(9.5, 9))
    for r, ch in enumerate(CHANNELS):
        err = np.abs(pred[r] - truth[r])
        vmax = np.nanmax(np.abs(masked(truth[r], mask)))
        err_vmax = np.nanmax(masked(err, mask))
        im_f, im_e = _panel_row(axs, r, [truth[r], pred[r], err], mask, vmax, err_vmax)
        axs[r, 0].set_ylabel(ch, fontsize=13, rotation=0, labelpad=15, va='center')
        fig.colorbar(im_f, ax=axs[r, :2].tolist(), shrink=0.85, pad=0.02)
        fig.colorbar(im_e, ax=axs[r, 2], shrink=0.85, pad=0.04)
    axs[0, 0].set_title('CFD truth')
    axs[0, 1].set_title('Model prediction')
    axs[0, 2].set_title('|error|')
    fig.suptitle(f'Reconstruction, test sample {idx} '
                 f'(Re = {sample["re"].item():.0f})', y=0.98)
    _save(fig, outdir, f'reconstruction_{idx}')


def find_transfer_triple(dataset):
    """Return (i, k, m) maximizing the Re jump: re_m == re_i, geo_m == geo_k
    != geo_i, re_k != re_i."""
    re, gid = dataset.re.tolist(), dataset.geo_ids.tolist()
    lookup = {}
    for m, (r, g) in enumerate(zip(re, gid)):
        lookup.setdefault((r, g), m)
    members = {}
    for k, g in enumerate(gid):
        members.setdefault(g, []).append(k)

    best, best_score = None, -1.0
    for m, (r_m, g_m) in enumerate(zip(re, gid)):
        i = next((j for j, (r_j, g_j) in enumerate(zip(re, gid))
                  if r_j == r_m and g_j != g_m), None)
        if i is None:
            continue
        for k in members[g_m]:
            if re[k] == r_m:
                continue
            score = abs(np.log10(re[k] / r_m))
            if score > best_score:
                best, best_score = (i, k, m), score
    return best


@torch.no_grad()
def transfer_figure(model, dataset, outdir):
    triple = find_transfer_triple(dataset)
    if triple is None:
        print('No cross-Re transfer triple found in this split; skipping figure B.')
        return
    i, k, m = triple
    re_i, re_k = dataset.re[i].item(), dataset.re[k].item()

    z_mu_i, _, _ = model.encode(dataset.fields[i:i + 1], dataset.sdf[i:i + 1])
    _, z_g_k, z_xi_k = model.encode(dataset.fields[k:k + 1], dataset.sdf[k:k + 1])
    pred = model.decoder(torch.cat([z_mu_i, z_g_k, z_xi_k], dim=1))[0].numpy()

    donor = dataset.fields[k].numpy()
    truth = dataset.fields[m].numpy()
    mask = dataset.mask[m, 0].numpy()

    fig, axs = plt.subplots(3, 4, figsize=(12.5, 9))
    for r, ch in enumerate(CHANNELS):
        err = np.abs(pred[r] - truth[r])
        vmax = max(np.nanmax(np.abs(masked(truth[r], mask))),
                   np.nanmax(np.abs(masked(donor[r], mask))))
        err_vmax = np.nanmax(masked(err, mask))
        im_f, im_e = _panel_row(axs, r, [donor[r], pred[r], truth[r], err],
                                mask, vmax, err_vmax)
        axs[r, 0].set_ylabel(ch, fontsize=13, rotation=0, labelpad=15, va='center')
        fig.colorbar(im_f, ax=axs[r, :3].tolist(), shrink=0.85, pad=0.02)
        fig.colorbar(im_e, ax=axs[r, 3], shrink=0.85, pad=0.04)
    axs[0, 0].set_title(f'donor: geometry\nat Re = {re_k:.0f}', fontsize=10)
    axs[0, 1].set_title(f'prediction\nat Re = {re_i:.0f}', fontsize=10)
    axs[0, 2].set_title(f'CFD truth\nat Re = {re_i:.0f}', fontsize=10)
    axs[0, 3].set_title('|error|', fontsize=10)
    fig.suptitle('Cross-Re transfer: the regime code moves a geometry '
                 f'from Re = {re_k:.0f} to Re = {re_i:.0f}', y=0.98)
    _save(fig, outdir, 'transfer')


def main(config_path, checkpoint_path, outdir, n_recon):
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
    os.makedirs(outdir, exist_ok=True)

    rng = np.random.default_rng(0)
    for idx in rng.choice(len(dataset), size=min(n_recon, len(dataset)), replace=False):
        reconstruction_figure(model, dataset, int(idx), outdir)
    transfer_figure(model, dataset, outdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate paper figures.')
    parser.add_argument('--config', type=str, default='configs/compositional/run7.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--outdir', type=str, default='docs/figures')
    parser.add_argument('--n-recon', type=int, default=2)
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.outdir, args.n_recon)

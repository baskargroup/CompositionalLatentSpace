import hashlib
import random
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler


class CompositionalLDCDataset(Dataset):
    """
    FlowBench 2D lid-driven cavity (steady flow with an object inside the cavity).

    Expects the FlowBench .npz layout:
        x: [num_samples, (Re, SDF, mask), 512, 512]
        y: [num_samples, (u, v, p, c_d, c_l), 512, 512]

    Each sample is returned as a dict with:
        fields   : (3, H, W)  target flow field (u, v, p)
        sdf      : (1, H, W)  signed distance field of the object (positive in fluid)
        mask     : (1, H, W)  binary fluid mask (1 = fluid, 0 = solid)
        re       : ()         raw Reynolds number
        log_re   : ()         standardized log10(Re), the regime label for z_mu
        geo_id   : ()         integer id of the geometry (samples sharing a shape
                              share an id), used by the same-factor invariance loss
    """

    def __init__(self, file_path_x, file_path_y, resolution=256, re_stats=None):
        x = np.load(file_path_x)['data']
        y = np.load(file_path_y)['data']

        # Re channel is a constant-valued map; extract the scalar
        re = torch.tensor(x[:, 0, 0, 0].copy(), dtype=torch.float32)

        # Group samples by geometry: identical shapes have byte-identical raw SDFs
        digests = [hashlib.md5(x[i, 1].tobytes()).hexdigest() for i in range(x.shape[0])]
        digest_to_id = {}
        geo_ids = []
        for d in digests:
            if d not in digest_to_id:
                digest_to_id[d] = len(digest_to_id)
            geo_ids.append(digest_to_id[d])
        self.geo_ids = torch.tensor(geo_ids, dtype=torch.long)

        sdf = torch.tensor(x[:, 1:2], dtype=torch.float32)
        mask = torch.tensor(x[:, 2:3], dtype=torch.float32)
        if mask.max() > 1.0:
            mask = mask / 255.0
        fields = torch.tensor(y[:, :3], dtype=torch.float32)  # u, v, p only

        if resolution is not None and resolution != fields.shape[-1]:
            size = (resolution, resolution)
            fields = F.interpolate(fields, size=size, mode='bilinear', align_corners=False)
            sdf = F.interpolate(sdf, size=size, mode='bilinear', align_corners=False)
            mask = F.interpolate(mask, size=size, mode='nearest')

        self.fields = fields
        self.sdf = sdf
        self.mask = mask
        self.re = re

        # Standardize log10(Re) with train-set statistics (pass them to the test set)
        log_re = torch.log10(re.clamp(min=1e-6))
        if re_stats is None:
            self.re_stats = (log_re.mean().item(), log_re.std().clamp(min=1e-8).item())
        else:
            self.re_stats = re_stats
        self.log_re = (log_re - self.re_stats[0]) / self.re_stats[1]

    def __len__(self):
        return self.fields.shape[0]

    def __getitem__(self, idx):
        return {
            'fields': self.fields[idx],
            'sdf': self.sdf[idx],
            'mask': self.mask[idx],
            're': self.re[idx],
            'log_re': self.log_re[idx],
            'geo_id': self.geo_ids[idx],
        }

    def geometry_descriptors(self):
        """
        Simple per-sample geometry summaries computed from the mask, used as
        probe targets for the z_g block: solid area fraction and solid centroid.
        Returns a (N, 3) tensor: [area_fraction, centroid_x, centroid_y].
        """
        solid = 1.0 - self.mask[:, 0]  # (N, H, W), 1 inside the object
        n, h, w = solid.shape
        area = solid.sum(dim=(1, 2)).clamp(min=1.0)
        xs = torch.linspace(0, 1, w).view(1, 1, w)
        ys = torch.linspace(0, 1, h).view(1, h, 1)
        cx = (solid * xs).sum(dim=(1, 2)) / area
        cy = (solid * ys).sum(dim=(1, 2)) / area
        return torch.stack([area / (h * w), cx, cy], dim=1)


class GroupedBatchSampler(Sampler):
    """
    Group-structured minibatches (working notes §13): every batch is composed
    of `groups_per_batch` geometry groups with `batch_size / groups_per_batch`
    samples each, so that samples sharing a geometry (at different Re) co-occur
    in the batch and the same-factor invariance loss (L10) is well-defined.
    """

    def __init__(self, group_ids, batch_size, groups_per_batch=4, seed=0):
        if batch_size % groups_per_batch != 0:
            raise ValueError('batch_size must be divisible by groups_per_batch')
        self.samples_per_group = batch_size // groups_per_batch
        self.groups_per_batch = groups_per_batch
        self.seed = seed
        self.epoch = 0

        self.groups = defaultdict(list)
        for idx, gid in enumerate(group_ids):
            self.groups[int(gid)].append(idx)

        max_group = max(len(v) for v in self.groups.values())
        if max_group < 2:
            warnings.warn('Every geometry appears only once in this dataset; '
                          'the same-factor invariance loss will be zero.')

    def _chunks(self, rng):
        chunks = []
        for idxs in self.groups.values():
            idxs = idxs[:]
            rng.shuffle(idxs)
            for j in range(0, len(idxs), self.samples_per_group):
                chunk = idxs[j:j + self.samples_per_group]
                while len(chunk) < self.samples_per_group:
                    chunk.append(rng.choice(idxs))  # pad short chunks from the group
                chunks.append(chunk)
        rng.shuffle(chunks)
        return chunks

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        chunks = self._chunks(rng)
        for b in range(len(chunks) // self.groups_per_batch):
            batch = []
            for chunk in chunks[b * self.groups_per_batch:(b + 1) * self.groups_per_batch]:
                batch.extend(chunk)
            yield batch

    def __len__(self):
        n_chunks = sum(-(-len(v) // self.samples_per_group) for v in self.groups.values())
        return n_chunks // self.groups_per_batch

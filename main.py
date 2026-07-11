import argparse
import os
import random
import time

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils import data

from data.dataset import CompositionalLDCDataset, GroupedBatchSampler
from models.compositional.compositional_ae import CompositionalAE


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)


def main(config_path):
    torch.set_float32_matmul_precision('high')
    config = OmegaConf.load(config_path)
    set_seed(config.trainer.seed)

    train_dataset = CompositionalLDCDataset(
        file_path_x=config.data.file_path_train_x,
        file_path_y=config.data.file_path_train_y,
        resolution=config.data.resolution,
    )
    test_dataset = CompositionalLDCDataset(
        file_path_x=config.data.file_path_test_x,
        file_path_y=config.data.file_path_test_y,
        resolution=config.data.resolution,
        re_stats=train_dataset.re_stats,  # standardize Re with train statistics
    )

    if config.model.get('lambda_swap', 0) > 0:
        # swap consistency needs same-Re pairs across different geometries
        re_to_geos = {}
        for re_val, gid in zip(train_dataset.re.tolist(),
                               train_dataset.geo_ids.tolist()):
            re_to_geos.setdefault(re_val, set()).add(gid)
        n_swappable = sum(1 for re_val, gid in zip(train_dataset.re.tolist(),
                                                   train_dataset.geo_ids.tolist())
                          if len(re_to_geos[re_val]) >= 2)
        print(f'Swap consistency: {n_swappable}/{len(train_dataset)} train samples '
              f'have a same-Re partner at a different geometry.')
        if n_swappable == 0:
            print('WARNING: no swappable pairs exist; the swap loss will be zero. '
                  'Re values likely differ across geometries.')

    if config.model.get('lambda_inv', 0) > 0:
        # group-structured minibatches: same geometry at several Re per batch,
        # required for the same-factor invariance loss (L10)
        sampler = GroupedBatchSampler(train_dataset.geo_ids.tolist(),
                                      batch_size=config.data.batch_size,
                                      groups_per_batch=config.data.groups_per_batch,
                                      seed=config.trainer.seed)
        train_loader = data.DataLoader(train_dataset, batch_sampler=sampler,
                                       num_workers=config.data.num_workers)
    else:
        train_loader = data.DataLoader(train_dataset, batch_size=config.data.batch_size,
                                       shuffle=True, drop_last=False,
                                       num_workers=config.data.num_workers)
    val_loader = data.DataLoader(test_dataset, batch_size=config.data.batch_size,
                                 shuffle=False, drop_last=False,
                                 num_workers=config.data.num_workers)

    model = CompositionalAE(resolution=config.data.resolution,
                            **OmegaConf.to_container(config.model, resolve=True))

    if config.trainer.get('wandb', False):
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(project=config.trainer.project, config=OmegaConf.to_container(config, resolve=True))
        run_name = logger.experiment.id
    else:
        logger = CSVLogger(save_dir='./logs', name=config.trainer.project)
        run_name = f'version_{logger.version}'

    checkpoint_dir = os.path.join(config.callbacks.checkpoint.dirpath, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        monitor=config.callbacks.checkpoint.monitor,
        dirpath=checkpoint_dir,
        filename=config.callbacks.checkpoint.filename,
        save_top_k=config.callbacks.checkpoint.save_top_k,
        mode=config.callbacks.checkpoint.mode,
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=config.trainer.max_epochs,
        callbacks=[checkpoint_callback],
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        log_every_n_steps=config.trainer.log_every_n_steps,
        logger=logger,
    )

    start_time = time.time()
    trainer.fit(model, train_loader, val_loader)
    print(f'Total training time: {time.time() - start_time:.2f} seconds')
    print(f'Best checkpoint: {checkpoint_callback.best_model_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the compositional autoencoder.')
    parser.add_argument('--config', type=str, default='configs/compositional/conf.yaml')
    args = parser.parse_args()
    main(args.config)

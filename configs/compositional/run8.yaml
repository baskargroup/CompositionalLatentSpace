data:
  # FlowBench 2D LDC (NS) .npz files
  file_path_train_x: /work/mech-ai/arabeh/data_prep_ns/LDC_NS_2D/processed/easy/all_ldc_train_x.npz
  file_path_train_y: /work/mech-ai/arabeh/data_prep_ns/LDC_NS_2D/processed/easy/all_ldc_train_y.npz
  file_path_test_x: /work/mech-ai/arabeh/data_prep_ns/LDC_NS_2D/processed/easy/all_ldc_test_x.npz
  file_path_test_y: /work/mech-ai/arabeh/data_prep_ns/LDC_NS_2D/processed/easy/all_ldc_test_y.npz
  resolution: 256
  batch_size: 16
  groups_per_batch: 4
  num_workers: 6

# Run-8 configuration: Run 7 + boundary-layer weighted reconstruction (L3).
# Motivated by the FlowBench comparison: our M2 (boundary-layer score) is flat
# at the M1 level while every baseline has M2 > M1. lambda_bl = 1.0 gives the
# thin 0 <= SDF <= 0.2 band the same total weight as the whole fluid region,
# i.e. each band pixel counts roughly fluid/band ~ 20x more than under L1 alone.
model:
  base_channels: 32
  latent_mu: 4
  latent_g: 32
  latent_xi: 16
  sdf_resolution: 64
  static_geometry: true
  lambda_recon: 1.0
  lambda_bl: 1.0
  lambda_regime: 0.1
  lambda_geo: 0.1
  lambda_decorr: 0.01
  lambda_inv: 0.1
  lambda_swap: 0.1
  lambda_xswap: 0.1
  lr: 0.001

trainer:
  project: compositional-run8
  wandb: false
  accelerator: auto
  devices: 1
  max_epochs: 200
  log_every_n_steps: 10
  seed: 0

callbacks:
  checkpoint:
    dirpath: ./checkpoints/compositional-run8
    filename: cae-{epoch:03d}
    monitor: val_loss_full
    mode: min
    save_top_k: 1

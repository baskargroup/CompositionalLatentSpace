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

# Run-7 configuration: Run 6 losses + static geometry encoder
# (z_g = SDFEncoder(sdf), Reynolds-invariant by construction — the decisive
# test of the flow-appearance mechanism found by the concept-vector diagnostic)
model:
  base_channels: 32
  latent_mu: 4
  latent_g: 32
  latent_xi: 16
  sdf_resolution: 64
  static_geometry: true
  lambda_recon: 1.0
  lambda_regime: 0.1
  lambda_geo: 0.1
  lambda_decorr: 0.01
  lambda_inv: 0.1
  lambda_swap: 0.1
  lambda_xswap: 0.1
  lr: 0.001

trainer:
  project: compositional-run7
  wandb: false
  accelerator: auto
  devices: 1
  max_epochs: 200
  log_every_n_steps: 10
  seed: 0

callbacks:
  checkpoint:
    dirpath: ./checkpoints/compositional-run7
    filename: cae-{epoch:03d}
    monitor: val_loss_full
    mode: min
    save_top_k: 1

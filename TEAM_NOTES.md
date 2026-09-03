# Local DP3 Extensions

This repository is based on
[`YanjieZe/3D-Diffusion-Policy`](https://github.com/YanjieZe/3D-Diffusion-Policy).
The local work in this branch adds the following code paths:

- SMP model components under
  `3D-Diffusion-Policy/diffusion_policy_3d/model/smp/`.
- Dense and adaptive SMP-DP3 policies under
  `3D-Diffusion-Policy/diffusion_policy_3d/policy/`.
- A multi-task DexArt dataset and runner for faucet, bucket, laptop, and toilet.
- A `torchrun`-based multi-GPU training entry point.
- Multi-seed DexArt evaluation, checkpoint selection, and evaluation scripts.

## Main Entry Points

- `3D-Diffusion-Policy/train_smp_dp3_multi_gpu.py`: distributed training.
- `3D-Diffusion-Policy/diffusion_policy_3d/config/smp_dp3.yaml`: base SMP-DP3
  configuration.
- `3D-Diffusion-Policy/diffusion_policy_3d/config/smp_dp3_adaptive_multi_gpu.yaml`:
  adaptive multi-GPU configuration.
- `scripts/train_smp_dp3_multi_gpu_extension.sh`: validated launcher for a list
  of CUDA device IDs.
- `scripts/evaluate_dexart_robust.sh`: best/latest checkpoint evaluation for the
  original DexArt runs.
- `scripts/evaluate_dexart_new_data.sh`: comparison evaluation for the new
  faucet and bucket datasets.

Run the multi-GPU launcher from the repository root after activating the DP3
environment:

```bash
bash scripts/train_smp_dp3_multi_gpu_extension.sh 0,1
```

The launcher accepts optional arguments in this order:

```text
GPU_LIST NUM_EXPERTS MASS_THRESHOLD PER_GPU_BATCH SEED OUTPUT_DIR NUM_EPOCHS RESUME
```

`MUJOCO_DIR` can override the default MuJoCo location
`$HOME/.mujoco/mujoco210`.

## Data And Runtime Files

The Git repository intentionally excludes `data/`, checkpoints, generated
videos, Python caches, and experiment output. The virtual environment, UV
cache, downloaded Python runtime, and local NVIDIA runtime are also outside
this repository. Follow `INSTALL.md` to create an environment, then provide the
datasets at the paths declared in the task YAML files. In particular,
`dexart_multitask_new_bucket.yaml` references:

```text
data/dexart_faucet_expert.zarr
data/dexart_bucket_new.zarr
data/dexart_laptop_expert.zarr
data/dexart_toilet_expert.zarr
```

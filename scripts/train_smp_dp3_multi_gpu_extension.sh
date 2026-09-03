#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 8 ]]; then
    echo "Usage: $0 GPU_LIST [NUM_EXPERTS] [MASS_THRESHOLD] [PER_GPU_BATCH] [SEED] [OUTPUT_DIR] [NUM_EPOCHS] [RESUME]" >&2
    exit 2
fi

gpu_list=$1
num_experts=${2:-8}
mass_threshold=${3:-0.95}
per_gpu_batch=${4:-32}
seed=${5:-42}
output_dir=${6:-data/outputs/dexart-smp-expert8}
num_epochs=${7:-1500}
resume=${8:-false}

IFS=',' read -r -a gpu_ids <<< "$gpu_list"
if [[ ${#gpu_ids[@]} -lt 1 ]]; then
    echo "GPU_LIST must contain at least one CUDA device" >&2
    exit 2
fi
for gpu_id in "${gpu_ids[@]}"; do
    if [[ ! $gpu_id =~ ^[0-9]+$ ]]; then
        echo "Invalid GPU id: $gpu_id" >&2
        exit 2
    fi
done
if [[ ! $num_experts =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_EXPERTS must be a positive integer" >&2
    exit 2
fi
if [[ ! $per_gpu_batch =~ ^[1-9][0-9]*$ ]]; then
    echo "PER_GPU_BATCH must be a positive integer" >&2
    exit 2
fi
if [[ ! $num_epochs =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_EPOCHS must be a positive integer" >&2
    exit 2
fi
if [[ $resume != true && $resume != false ]]; then
    echo "RESUME must be true or false" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_dir=$(cd -- "$script_dir/.." && pwd)
code_dir="$repository_dir/3D-Diffusion-Policy"
workspace_dir=$(cd -- "$repository_dir/.." && pwd)

driver_version_line=$(< /proc/driver/nvidia/version)
if [[ $driver_version_line =~ Kernel[[:space:]]Module[[:space:]]+([0-9.]+) ]]; then
    driver_runtime_dir="$workspace_dir/.runtime/nvidia-${BASH_REMATCH[1]}"
    if [[ -d $driver_runtime_dir ]]; then
        export LD_LIBRARY_PATH="$driver_runtime_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
fi

mujoco_dir=${MUJOCO_DIR:-${HOME}/.mujoco/mujoco210}
if [[ -d $mujoco_dir/bin ]]; then
    export LD_LIBRARY_PATH="$mujoco_dir/bin:/usr/lib/nvidia${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export MUJOCO_GL=egl
    export MUJOCO_PY_MUJOCO_PATH=$mujoco_dir
fi

if command -v torchrun >/dev/null 2>&1; then
    torchrun_bin=$(command -v torchrun)
elif [[ -x "$workspace_dir/.venv/bin/torchrun" ]]; then
    torchrun_bin="$workspace_dir/.venv/bin/torchrun"
else
    echo "torchrun was not found; activate the DP3 environment first" >&2
    exit 127
fi

cd "$code_dir"
export CUDA_VISIBLE_DEVICES=$gpu_list
export HYDRA_FULL_ERROR=1
# NCCL 2.18 probes NVLS on RTX 4090 systems and can fail during communicator init.
export NCCL_NVLS_ENABLE=0

exec "$torchrun_bin" \
    --standalone \
    --nproc_per_node="${#gpu_ids[@]}" \
    train_smp_dp3_multi_gpu.py \
    --config-name=smp_dp3_adaptive_multi_gpu.yaml \
    "smp.num_experts=$num_experts" \
    "smp.inference.mass_threshold=$mass_threshold" \
    "dataloader.batch_size=$per_gpu_batch" \
    "training.num_epochs=$num_epochs" \
    "training.resume=$resume" \
    "training.seed=$seed" \
    "logging.mode=disabled" \
    "hydra/job_logging=disabled" \
    "distributed.output_dir=$output_dir" \
    "exp_name=dexart-smp-expert8"

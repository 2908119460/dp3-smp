#!/usr/bin/env bash
set -uo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/.." && pwd)
code_root=${repo_root}/3D-Diffusion-Policy
dp3_root=$(cd "${repo_root}/.." && pwd)
result_root=${code_root}/data/outputs
summary_dir=${result_root}/dexart_new_dp3_repro0813_comparison
driver_root=${DP3_DRIVER_ROOT:-${dp3_root}/.runtime/nvidia-580.126.09}
interval=${EVAL_POLL_INTERVAL:-60}

mkdir -p "${summary_dir}"
exec 9>"${summary_dir}/queue.lock"
if ! flock -n 9; then
    printf 'Another new-data evaluation queue is already running.\n' >&2
    exit 1
fi

printf '%s\n' "$$" >"${summary_dir}/queue.pid"
trap 'rm -f "${summary_dir}/queue.pid"' EXIT

source "${dp3_root}/activate_dp3.sh"
export LD_PRELOAD="${driver_root}/libcuda.so.580.126.09:${driver_root}/libnvidia-ml.so.580.126.09"
export LD_LIBRARY_PATH="${driver_root}:${LD_LIBRARY_PATH}"
export VK_DRIVER_FILES="${driver_root}/nvidia_icd_local.json"
export VK_ICD_FILENAMES="${driver_root}/nvidia_icd_local.json"
export VK_LOADER_LAYERS_DISABLE='~implicit~'
export __EGL_VENDOR_LIBRARY_FILENAMES="${driver_root}/egl_vendor_local.json"
export __EGL_EXTERNAL_PLATFORM_CONFIG_DIRS="${dp3_root}/.runtime/empty-egl-platforms"
export WANDB_SILENT=true
export HYDRA_FULL_ERROR=1

tasks=(faucet bucket)
checkpoint_kinds=(best latest)

log_message() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

parse_metric() {
    local pattern=$1
    local log=$2
    rg -a -o "${pattern}: [0-9.]+" "${log}" 2>/dev/null \
        | tail -1 | awk '{print $2}'
}

write_summary() {
    local csv=${summary_dir}/results.csv
    local md=${summary_dir}/README.md
    local task kind original_run new_run original_log new_log status
    local original_rate new_rate delta std ci_low ci_high checkpoint_path

    printf 'task,checkpoint_kind,original_success_rate,new_success_rate,delta,new_seed_std,new_ci95_low,new_ci95_high,episodes,status,checkpoint_path\n' >"${csv}"
    printf '# DexArt new-data comparison\n\n' >"${md}"
    printf 'Original and new DP3 checkpoints are each evaluated with five seeds and 20 rollouts per seed.\n\n' >>"${md}"
    printf '| Task | Checkpoint | Original SR | New SR | Delta | New seed std | New 95%% CI | Status |\n' >>"${md}"
    printf '|---|---|---:|---:|---:|---:|---:|---|\n' >>"${md}"

    for task in "${tasks[@]}"; do
        original_run=${result_root}/dexart_${task}-dp3-repro0813_seed0
        new_run=${result_root}/dexart_new_${task}-dp3-repro0813_seed0
        for kind in "${checkpoint_kinds[@]}"; do
            original_log=${original_run}/robust_eval_${kind}.log
            new_log=${new_run}/robust_eval_${kind}.log
            status=pending
            [[ -e "${new_run}/robust_eval_${kind}.complete" ]] && status=complete
            [[ -e "${new_run}/robust_eval_${kind}.failed" ]] && status=failed
            [[ -e "${new_run}/evaluation.pipeline_failed" ]] && status=training_failed

            original_rate=$(parse_metric test_mean_score "${original_log}")
            new_rate=$(parse_metric test_mean_score "${new_log}")
            std=$(parse_metric eval_success_rate_seed_std "${new_log}")
            ci_low=$(parse_metric eval_success_rate_ci95_low "${new_log}")
            ci_high=$(parse_metric eval_success_rate_ci95_high "${new_log}")
            checkpoint_path=$(rg -a -o 'Resuming from checkpoint .*' "${new_log}" 2>/dev/null \
                | tail -1 | sed 's/^Resuming from checkpoint //')
            delta=NA
            if [[ -n "${original_rate}" && -n "${new_rate}" ]]; then
                delta=$(awk -v new="${new_rate}" -v old="${original_rate}" \
                    'BEGIN { printf "%.4f", new - old }')
            fi

            printf '%s,%s,%s,%s,%s,%s,%s,%s,100,%s,%s\n' \
                "${task}" "${kind}" "${original_rate:-NA}" "${new_rate:-NA}" \
                "${delta}" "${std:-NA}" "${ci_low:-NA}" "${ci_high:-NA}" \
                "${status}" "${checkpoint_path:-NA}" >>"${csv}"
            printf '| %s | %s | %s | %s | %s | %s | %s-%s | %s |\n' \
                "${task}" "${kind}" "${original_rate:-NA}" "${new_rate:-NA}" \
                "${delta}" "${std:-NA}" "${ci_low:-NA}" "${ci_high:-NA}" \
                "${status}" >>"${md}"
        done
    done
}

evaluate_task() {
    local task=$1
    local gpu=$2
    local run_dir=${result_root}/dexart_new_${task}-dp3-repro0813_seed0
    local kind log complete_marker failed_marker rc

    while [[ ! -e "${run_dir}/training.complete" ]]; do
        if [[ -e "${run_dir}/training.failed" ]]; then
            touch "${run_dir}/evaluation.pipeline_failed"
            log_message "Training failed for ${task}; evaluation will not run."
            return 1
        fi
        sleep "${interval}"
    done

    for kind in "${checkpoint_kinds[@]}"; do
        log=${run_dir}/robust_eval_${kind}.log
        complete_marker=${run_dir}/robust_eval_${kind}.complete
        failed_marker=${run_dir}/robust_eval_${kind}.failed
        if [[ -e "${complete_marker}" ]]; then
            log_message "Skipping completed evaluation: ${task} ${kind}."
            continue
        fi

        rm -f "${failed_marker}"
        log_message "Starting ${task} ${kind} evaluation on GPU ${gpu}."
        if (
            cd "${code_root}"
            export CUDA_VISIBLE_DEVICES=${gpu}
            python -u eval.py --config-name=dp3.yaml \
                task=dexart_new_${task} \
                hydra.run.dir="${run_dir}" \
                training.seed=0 \
                training.device=cuda:0 \
                exp_name=dexart_new_${task}-dp3-repro0813 \
                logging.mode=offline \
                logging.name=robust-eval-${kind} \
                checkpoint.save_ckpt=true \
                evaluation.checkpoint=${kind} \
                >"${log}" 2>&1
        ); then
            touch "${complete_marker}"
            log_message "Completed ${task} ${kind} evaluation."
        else
            rc=$?
            printf '%s\n' "${rc}" >"${failed_marker}"
            log_message "Failed ${task} ${kind} evaluation with exit code ${rc}."
            return "${rc}"
        fi
    done
}

write_summary
evaluate_task faucet 5 &
faucet_pid=$!
evaluate_task bucket 7 &
bucket_pid=$!

pipeline_rc=0
wait "${faucet_pid}" || pipeline_rc=1
wait "${bucket_pid}" || pipeline_rc=1
write_summary

if (( pipeline_rc == 0 )); then
    touch "${summary_dir}/evaluation.complete"
    log_message "All evaluations completed: ${summary_dir}/results.csv"
else
    touch "${summary_dir}/evaluation.failed"
    log_message 'One or more training/evaluation jobs failed.'
fi
exit "${pipeline_rc}"

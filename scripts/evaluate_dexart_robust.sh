#!/usr/bin/env bash
set -uo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/.." && pwd)
code_root=${repo_root}/3D-Diffusion-Policy
dp3_root=$(cd "${repo_root}/.." && pwd)
result_root=${code_root}/data/outputs
summary_dir=${result_root}/dexart_dp3_repro0813_robust_eval
interval=${GPU_POLL_INTERVAL:-60}
min_free_mib=${GPU_MIN_FREE_MIB:-30000}
max_utilization=${GPU_MAX_UTILIZATION:-10}
stable_checks=${GPU_STABLE_CHECKS:-3}

mkdir -p "${summary_dir}"
exec 9>"${summary_dir}/queue.lock"
if ! flock -n 9; then
    printf 'Another robust evaluation queue is already running.\n' >&2
    exit 1
fi

printf '%s\n' "$$" >"${summary_dir}/queue.pid"
trap 'rm -f "${summary_dir}/queue.pid"' EXIT

source "${dp3_root}/activate_dp3.sh"
export WANDB_SILENT=true
export HYDRA_FULL_ERROR=1

tasks=(faucet toilet bucket)
checkpoint_kinds=(best latest)

log_message() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

all_training_complete() {
    local task run_dir
    for task in "${tasks[@]}"; do
        run_dir=${result_root}/dexart_${task}-dp3-repro0813_seed0
        [[ -e "${run_dir}/training.complete" ]] || return 1
    done
}

find_idle_gpu() {
    nvidia-smi \
        --query-gpu=index,memory.free,utilization.gpu \
        --format=csv,noheader,nounits \
        | awk -F, -v min_free="${min_free_mib}" -v max_util="${max_utilization}" '
            {
                gsub(/ /, "", $1)
                gsub(/ /, "", $2)
                gsub(/ /, "", $3)
                if ($2 >= min_free && $3 <= max_util) {
                    print $1
                    exit
                }
            }
        '
}

wait_for_idle_gpu() {
    local candidate previous='' consecutive=0
    while true; do
        candidate=$(find_idle_gpu)
        if [[ -n "${candidate}" && "${candidate}" == "${previous}" ]]; then
            consecutive=$((consecutive + 1))
        elif [[ -n "${candidate}" ]]; then
            previous=${candidate}
            consecutive=1
        else
            previous=''
            consecutive=0
        fi

        if (( consecutive >= stable_checks )); then
            printf '%s\n' "${candidate}"
            return 0
        fi
        sleep "${interval}"
    done
}

write_summary() {
    local csv=${summary_dir}/results.csv
    local md=${summary_dir}/README.md
    local task kind run_dir log status checkpoint_path periodic_score rate std ci_low ci_high

    printf 'task,checkpoint_kind,checkpoint_path,periodic_score,robust_success_rate,seed_std,ci95_low,ci95_high,episodes,status\n' >"${csv}"
    printf '# DexArt robust evaluation\n\n' >"${md}"
    printf 'Five evaluation seeds, 20 rollouts per seed, 100 rollouts per checkpoint.\n\n' >>"${md}"
    printf '| Task | Checkpoint | Periodic SR | Robust SR | Seed std | 95%% CI | Status |\n' >>"${md}"
    printf '|---|---|---:|---:|---:|---:|---|\n' >>"${md}"

    for task in "${tasks[@]}"; do
        run_dir=${result_root}/dexart_${task}-dp3-repro0813_seed0
        for kind in "${checkpoint_kinds[@]}"; do
            log=${run_dir}/robust_eval_${kind}.log
            status=pending
            [[ -e "${run_dir}/robust_eval_${kind}.complete" ]] && status=complete
            [[ -e "${run_dir}/robust_eval_${kind}.failed" ]] && status=failed
            checkpoint_path=$(rg -a -o 'Resuming from checkpoint .*' "${log}" 2>/dev/null | tail -1 | sed 's/^Resuming from checkpoint //')
            periodic_score=$(printf '%s\n' "${checkpoint_path}" | sed -n 's/.*test_mean_score=\([0-9.]*\)\.ckpt/\1/p')
            [[ "${kind}" == latest ]] && periodic_score=NA
            rate=$(rg -a -o 'test_mean_score: [0-9.]+' "${log}" 2>/dev/null | tail -1 | awk '{print $2}')
            std=$(rg -a -o 'eval_success_rate_seed_std: [0-9.]+' "${log}" 2>/dev/null | tail -1 | awk '{print $2}')
            ci_low=$(rg -a -o 'eval_success_rate_ci95_low: [0-9.]+' "${log}" 2>/dev/null | tail -1 | awk '{print $2}')
            ci_high=$(rg -a -o 'eval_success_rate_ci95_high: [0-9.]+' "${log}" 2>/dev/null | tail -1 | awk '{print $2}')
            printf '%s,%s,%s,%s,%s,%s,%s,%s,100,%s\n' \
                "${task}" "${kind}" "${checkpoint_path:-NA}" \
                "${periodic_score:-NA}" "${rate:-NA}" "${std:-NA}" \
                "${ci_low:-NA}" "${ci_high:-NA}" "${status}" >>"${csv}"
            printf '| %s | %s | %s | %s | %s | %s-%s | %s |\n' \
                "${task}" "${kind}" "${periodic_score:-NA}" \
                "${rate:-NA}" "${std:-NA}" "${ci_low:-NA}" \
                "${ci_high:-NA}" "${status}" >>"${md}"
        done
    done
}

write_summary
while ! all_training_complete; do
    log_message 'Waiting for all three DexArt training runs to complete.'
    sleep "${interval}"
done

for task in "${tasks[@]}"; do
    run_dir=${result_root}/dexart_${task}-dp3-repro0813_seed0
    for kind in "${checkpoint_kinds[@]}"; do
        complete_marker=${run_dir}/robust_eval_${kind}.complete
        failed_marker=${run_dir}/robust_eval_${kind}.failed
        log=${run_dir}/robust_eval_${kind}.log
        if [[ -e "${complete_marker}" ]]; then
            log_message "Skipping completed evaluation: ${task} ${kind}."
            continue
        fi

        rm -f "${failed_marker}"
        gpu=$(wait_for_idle_gpu)
        log_message "Starting ${task} ${kind} evaluation on GPU ${gpu}."
        if (
            cd "${code_root}"
            export CUDA_VISIBLE_DEVICES=${gpu}
            python -u eval.py --config-name=dp3.yaml \
                task=dexart_${task} \
                hydra.run.dir="${run_dir}" \
                training.seed=0 \
                training.device=cuda:0 \
                exp_name=dexart_${task}-dp3-repro0813 \
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
        fi
        write_summary
    done
done

write_summary
log_message "All robust evaluations processed. Results: ${summary_dir}/results.csv"

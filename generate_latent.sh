#!/bin/bash
# Generate the AE training latent corpus. Parallel sharding across 1–8 GPUs.
#
# The Python entry point (generate_latent.py) is already shard-aware: each
# process pins to one physical GPU via CUDA_VISIBLE_DEVICES (so it sees cuda:0)
# and gets a disjoint slice of the randomly sampled prompts via
# --gpu_id / --num_shards.
#
# Usage:
#   bash generate_latent.sh                       # default 8 GPUs (0..7)
#   NGPU=4 GPUS=0,1,2,3 bash generate_latent.sh   # 4 GPUs
#   NGPU=1 GPUS=2 bash generate_latent.sh         # single GPU on physical id 2
#   CONFIG=configs/other.yaml bash generate_latent.sh

set -uo pipefail

NGPU="${NGPU:-8}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
CONFIG="${CONFIG:-configs/generate_latent.yaml}"

IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
if (( ${#GPU_ARRAY[@]} < NGPU )); then
    echo "Need at least ${NGPU} GPU ids in GPUS='${GPUS}' (got ${#GPU_ARRAY[@]})" >&2
    exit 1
fi

echo "Launching ${NGPU} workers with config: ${CONFIG}"
pids=()
for ((i=0; i<NGPU; i++)); do
    phys_gpu="${GPU_ARRAY[$i]}"
    echo "  shard ${i}/${NGPU}  →  physical GPU ${phys_gpu}"
    CUDA_VISIBLE_DEVICES="${phys_gpu}" python generate_latent.py \
        --config_path "${CONFIG}" \
        --gpu_id "${i}" \
        --num_shards "${NGPU}" &
    pids+=($!)
done

# Ctrl-C / SIGTERM → kill every worker before exiting
trap 'echo; echo "Interrupted — killing workers"; kill "${pids[@]}" 2>/dev/null; exit 130' INT TERM

status=0
for pid in "${pids[@]}"; do
    wait "${pid}" || status=$?
done

if (( status == 0 )); then
    echo "All ${NGPU} shards finished successfully."
else
    echo "At least one shard failed (last non-zero status: ${status})." >&2
fi
exit ${status}

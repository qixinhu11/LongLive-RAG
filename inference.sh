#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# LongLive-RAG inference launcher.
#
# Usage:
#   bash inference.sh <backbone> <method>
#
# Backbones: causal_forcing | self_forcing | longlive
# Methods:   native | latentmem
#
# Examples:
#   bash inference.sh longlive latentmem         # main result (LongLive + RAG)
#   bash inference.sh causal_forcing native      # baseline, no retrieval
#   bash inference.sh self_forcing latentmem     # Self-Forcing + RAG
#
# Overrides:
#   GPU=4 PORT=29510 bash inference.sh longlive latentmem
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

BACKBONE="${1:-longlive}"
METHOD="${2:-latentmem}"
GPU="${GPU:-0}"
PORT="${PORT:-29501}"

CONFIG="configs/${BACKBONE}_${METHOD}.yaml"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found at $CONFIG"
    echo "  Valid backbones: causal_forcing | self_forcing | longlive"
    echo "  Valid methods:   native | latentmem"
    exit 1
fi

echo "════════════════════════════════════════════════════════════"
echo "  Backbone : ${BACKBONE}"
echo "  Method   : ${METHOD}"
echo "  Config   : ${CONFIG}"
echo "  GPU      : ${GPU}"
echo "════════════════════════════════════════════════════════════"

CUDA_VISIBLE_DEVICES="${GPU}" torchrun \
    --nproc_per_node=1 \
    --master_port="${PORT}" \
    inference.py \
        --config_path "${CONFIG}"

#!/bin/bash
# Train the LongLive-RAG retrieval autoencoder (single GPU).
# All hyperparameters live in ae/configs/ae_delta.yaml.
#
# Override the GPU via CUDA_VISIBLE_DEVICES if needed:
#   CUDA_VISIBLE_DEVICES=2 bash train_ae_delta.sh

set -euo pipefail

python -m ae.train --config ae/configs/ae_delta.yaml

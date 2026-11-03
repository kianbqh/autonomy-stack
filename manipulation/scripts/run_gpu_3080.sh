#!/usr/bin/env bash
set -euo pipefail
# Use an existing CUDA-enabled PyTorch environment. This wrapper does not install CUDA.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
exec bash "$(dirname "$0")/run_all.sh" --device cuda "$@"

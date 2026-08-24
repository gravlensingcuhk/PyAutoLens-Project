#!/bin/bash
# HTCondor job wrapper for the single no-subhalo baseline fit.
# Run this FIRST; it must complete before (or alongside) the tile jobs.

export CPUS_PER_TASK=1
export OPENBLAS_NUM_THREADS=$CPUS_PER_TASK
export MKL_NUM_THREADS=$CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$CPUS_PER_TASK
export OMP_NUM_THREADS=$CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$CPUS_PER_TASK

DATASET="$1"
FILT="${2:-F444W}"
SRC_PREFIX="${3:-model_setup_free_source}"
MASS_PREFIX="${4:-mass_models}"

exec uv run python prepare_subhalo_baseline.py "$DATASET" "$FILT" "$SRC_PREFIX" "$MASS_PREFIX"

#!/bin/bash
#
# Copy finished search archives from phygpu's output/ to the HTCondor cluster's project output,
# where proposed_subhalo/ reads source_pix[1] and mass_multipole by directory.
#
#   CONDOR_REMOTE=user@condor-host phygpu/sync_to_condor.sh [--dry-run] [lens ...]
#
# Run from phygpu (or anywhere holding output/). Only *.zip archives travel, in the same
# output/<prefix>/<lens>/<search>/<id>.zip layout: not Nautilus checkpoints (*.hdf5), not the
# `.running` lock directories of searches still fitting, not read-only replay folders
# (`*.replay-*`), and never anything deleted on the far side (no --delete). An archive is
# complete when it holds `.completed`, which every search writes before zipping, so a partial
# chain is safe to sync and sync again later.
#
# The CPU side must run the same commit of this repository (same vendored pyauto/ packages),
# built with a plain `uv sync`: see slam_pipeline/search_util.py "__CPU Replay__".

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
CONDOR_REMOTE=${CONDOR_REMOTE:?set CONDOR_REMOTE=user@host of the HTCondor cluster}
CONDOR_PROJECT=${CONDOR_PROJECT:-/users/wing-yan.chan/PyAutoLens-Project}

DRY=
if [ "${1:-}" = "--dry-run" ]; then DRY="-n -i"; shift; fi

LENSES=("$@")
[ ${#LENSES[@]} -gt 0 ] || LENSES=(COSJ100018+022138 COSJ100024+021749)

for lens in "${LENSES[@]}"; do
    for prefix in model_setup_free_source mass_models; do
        src="$PROJECT_ROOT/output/$prefix/$lens/"
        [ -d "$src" ] || { echo "skip $prefix/$lens (nothing here)"; continue; }
        echo "== $prefix/$lens"
        ssh "$CONDOR_REMOTE" "mkdir -p '$CONDOR_PROJECT/output/$prefix/$lens'"
        rsync -rlpt -v --info=skip0 $DRY \
            --include '*/' --include '*.zip' --exclude '*' \
            "$src" "$CONDOR_REMOTE:$CONDOR_PROJECT/output/$prefix/$lens/"
    done
done

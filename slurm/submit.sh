#!/usr/bin/env bash
# Submit a group of runs from the repository root, with $DATA and $OUTPUT set. At most 2 run at once.
#   bash slurm/submit.sh --datasets dtd eurosat --methods MaPLe CoCoOp --seeds 1
# Methods without training run once, whatever --seeds says. --force redoes finished runs.
set -euo pipefail

source slurm/activate_env.sh

# The repository root must be importable, because the script itself is in tools/.
PYTHONPATH="${PWD}" python tools/submit_runs.py "$@"

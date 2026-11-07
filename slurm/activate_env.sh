#!/usr/bin/env bash
# Repository-root check and conda activation shared by the job scripts.
# Source this file, do not run it.
set -euo pipefail

if [ ! -f train.py ]; then
  echo "ERROR: expected to run from the repository root, found no train.py in ${PWD}." >&2
  exit 1
fi

# conda's activation scripts read variables that may be unset.
set +u
source /apps/anaconda3/etc/profile.d/conda.sh
conda deactivate || true    # clear any env inherited from the submitting shell
conda activate maple
set -u

echo "python: $(which python)"
if [[ "$(which python)" != *"/envs/maple/"* ]]; then
  echo "ERROR: the maple environment is not active in this job." >&2
  exit 1
fi

echo "git commit: $(git rev-parse HEAD)"

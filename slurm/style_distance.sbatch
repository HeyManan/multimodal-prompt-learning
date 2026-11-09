#!/bin/bash -l
# Style distance to ImageNet: CLIP features of DomainNet and ImageNet images, then FD and
# nearest-neighbor distance per style. Submit from the repository root with $DATA and $OUTPUT set:
#   sbatch slurm/style_distance.sbatch
#   sbatch --time=00:10:00 slurm/style_distance.sbatch --smoke
# Features are cached in $OUTPUT/distance/features/, so a resubmitted job reuses them.
#
# ClsGenAI is the GAIVI class partition. Other clusters must change it.
#SBATCH --job-name=style_distance
#SBATCH --partition=ClsGenAI
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=style_distance_%j.out

set -euo pipefail

if [ "$#" -gt 1 ] || { [ "$#" -eq 1 ] && [ "$1" != "--smoke" ]; }; then
  echo "ERROR: expected no arguments or --smoke, found: $*" >&2
  exit 1
fi

source slurm/activate_env.sh
nvidia-smi

# The repository root must be importable, because the scripts themselves are in tools/.
export PYTHONPATH="${PWD}"

if [ "$#" -eq 0 ]; then
  python tools/distance/extract_features.py
  python tools/distance/compute_distances.py
else
  # The smoke run always recomputes its features, so a cached file cannot hide a broken step.
  python tools/distance/extract_features.py --smoke --force
  python tools/distance/compute_distances.py --smoke
  results_file="${OUTPUT}/smoke/distance/fd.csv"
  if [ ! -f "${results_file}" ]; then
    echo "ERROR: expected ${results_file}, found no such file." >&2
    exit 1
  fi
  cat "${results_file}"
  echo "Style distance smoke test passed."
fi

#!/bin/bash -l
# One (method, dataset, seed) run. Submit from the repository root with $DATA and $OUTPUT set:
#   sbatch slurm/run_one.sbatch --method MaPLe --dataset dtd --seed 1
# slurm/submit.sh submits it as an array and passes a task file with one run per line.
#
# ClsGenAI is the GAIVI class partition. Other clusters must change it.
#SBATCH --job-name=base2new
#SBATCH --partition=ClsGenAI
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=base2new_%j.out

set -euo pipefail

source slurm/activate_env.sh
nvidia-smi

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  read -r -a run_args < <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")
else
  run_args=("$@")
fi

# The repository root must be importable, because the script itself is in tools/.
PYTHONPATH="${PWD}" python tools/run_base2new.py "${run_args[@]}"

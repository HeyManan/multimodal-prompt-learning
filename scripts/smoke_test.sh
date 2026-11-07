#!/bin/bash -l
# End-to-end check: MaPLe on DTD, seed 1, 1 epoch, ending in a results.json.
# Submit from the repository root with $DATA and $OUTPUT set:
#   sbatch scripts/smoke_test.sh
#
# The 10 minute limit is the project's limit for this test, so a slower run fails.
# ClsGenAI is the GAIVI class partition. Other clusters must change it.
#SBATCH --job-name=smoke
#SBATCH --partition=ClsGenAI
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=smoke_%j.out

set -euo pipefail

source slurm/activate_env.sh
nvidia-smi

# The repository root must be importable, because the scripts themselves are in tools/.
export PYTHONPATH="${PWD}"
python tools/prepare_fewshot_cache.py --dataset dtd --seeds 1
python tools/run_base2new.py --method MaPLe --dataset dtd --seed 1 --smoke --force

results_file="${OUTPUT}/smoke/base2new/dtd/MaPLe/seed1/results.json"
if [ ! -f "${results_file}" ]; then
  echo "ERROR: expected ${results_file}, found no such file." >&2
  exit 1
fi
cat "${results_file}"
echo "Smoke test passed."

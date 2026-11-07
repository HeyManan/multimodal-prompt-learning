#!/bin/bash -l
# End-to-end check on DomainNet real with a small test subset: MaPLe, seed 1, 1 epoch, then
# ZeroshotCLIP on base and new, each ending in a results.json.
# Submit from the repository root with $DATA and $OUTPUT set:
#   sbatch scripts/smoke_test_domainnet.sh
#
# The 10 minute limit is the project's limit for this test, so a slower run fails.
# ClsGenAI is the GAIVI class partition. Other clusters must change it.
#SBATCH --job-name=smoke_domainnet
#SBATCH --partition=ClsGenAI
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=smoke_domainnet_%j.out

set -euo pipefail

source slurm/activate_env.sh
nvidia-smi

dataset="domainnet_b2n_real_smoke"

# The repository root must be importable, because the script itself is in tools/.
export PYTHONPATH="${PWD}"
python tools/run_base2new.py --method MaPLe --dataset "${dataset}" --seed 1 --smoke --force
python tools/run_base2new.py --method ZeroshotCLIP --dataset "${dataset}" --smoke --force

for run in MaPLe/seed1 ZeroshotCLIP/seednone; do
  results_file="${OUTPUT}/smoke/base2new/${dataset}/${run}/results.json"
  if [ ! -f "${results_file}" ]; then
    echo "ERROR: expected ${results_file}, found no such file." >&2
    exit 1
  fi
  cat "${results_file}"
done
echo "Smoke test passed."

Base: github.com/muzairkhattak/multimodal-prompt-learning @ 69bce21ae8eda80ad6187534b2dce09cf6c59e17

Dependency: github.com/KaiyangZhou/Dassl.pytorch @ c61a1b570ac6333bd50fb5ae06aea59002fb20bb

| Component | Origin | What I changed |
| --- | --- | --- |
| Training loop and trainers (MaPLe, IVLP, CoCoOp, ZeroshotCLIP) | Base repo @ 69bce21 | Unmodified |
| `.gitignore` | New | Keeps outputs, logs, datasets and checkpoints out of git |
| `tools/data/common.sh` | New | Download, checksum and image count helpers shared by the download scripts |
| `tools/data/download_domainnet.sh` | New | Downloads the 6 DomainNet styles and the cleaned train/test lists, prints image counts |
| `tools/data/download_eurosat_dtd_aircraft.sh` | New | Downloads EuroSAT, DTD and FGVC Aircraft into the `docs/DATASETS.md` layout; EuroSAT comes from the authors' Zenodo record because the documented host timed out |
| `tools/check_env.py` | New | Prints torch and CUDA versions, runs CLIP ViT-B/16 once on the GPU |
| `slurm/test_env.sbatch` | New | One GPU, 10 minute GAIVI job that runs the environment check |
| `requirements.txt` | Base repo @ 69bce21 | Replaced its 3 lines with every pip-installed version in `environment_freeze.txt`, plus `wheel`, which `tb-nightly` requires |
| `environment_freeze.txt` | New | `conda list` export of the environment on GAIVI |
| `README.md` | New | Project summary, pinned commits, GAIVI setup, smoke test and how to reproduce the results. The base repo's README is kept at `docs/ORIGINAL_README.md` |
| `tools/data/write_domainnet_notes.py` | New | Generates `data/README.md` from the DomainNet lists |
| `utils/domainnet.py` | New | DomainNet styles, the fixed 0-172 / 173-344 split, the class name rule and list reading |
| `tools/data/prepare_domainnet_splits.py` | New | Writes everything in `splits/domainnet/` from the official lists |
| `splits/domainnet/` | New | Generated: class list, per-class audit, 16-shot train and val files for seeds 1-3, and a fixed test subset per style |
| `data/README.md` | New | Generated notes: cleaned-list choice, fixed base/new split, classes missing from painting, classes with fewer than 16 training images |
| `utils/methods.py` | New | Method registry: trainer, config file and final epoch for MaPLe, CoCoOp and ZeroshotCLIP, plus the shot count and seeds |
| `utils/paths.py` | New | Reads `$DATA` and `$OUTPUT`, builds run, cache and DomainNet split paths, checks split and cache files exist |
| `utils/logging_setup.py` | New | Logging setup shared by the scripts; re-enables logging after an import disables it |
| `utils/log_parsing.py` | New | Reads correct and total counts from Dassl's `log.txt`, because the log prints accuracy to 1 decimal |
| `utils/run_record.py` | New | Accuracy, harmonic mean, git commit, Slurm job id and writing `results.json` |
| `utils/paper_results.py` | New | The MaPLe paper's Table 3 targets for DTD and EuroSAT |
| `tools/prepare_fewshot_cache.py` | New | Builds the repo's few-shot cache for one dataset and its seeds in one process, then checks the image counts |
| `tools/run_base2new.py` | New | Runs one (method, dataset, seed) by calling `train.py`, without changing it, and writes `results.json`; zero-shot runs `--eval-only` on base and new with a fixed RNG seed |
| `tools/submit_runs.py` | New | Expands a group into runs, skips finished ones, and submits a Slurm array limited to 2 jobs at once |
| `tools/analysis/collect_results.py` | New | Gathers every `results.json` into `results/runs.csv` |
| `tools/analysis/reproduction_table.py` | New | Writes the CSV and Markdown table comparing our results with the paper |
| `slurm/activate_env.sh` | New | Repository-root check and conda activation, with the rules from `slurm/test_env.sbatch` |
| `slurm/run_one.sbatch` | New | One GPU GAIVI job for one run; also runs as an array from a task file |
| `slurm/submit.sh` | New | Activates the environment and calls `tools/submit_runs.py` |
| `scripts/smoke_test.sh` | New | 10 minute end-to-end test: MaPLe on DTD, seed 1, 1 epoch |

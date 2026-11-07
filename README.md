# Prompt Learning on DomainNet Styles

Graduate class project for CIS 6930 Generative AI at USF. It tests MaPLe (Multi-modal Prompt
Learning, Khattak et al., CVPR 2023) and related prompt-learning methods on the DomainNet styles,
in the paper's base-to-novel setting: train on the base classes, then evaluate on the new classes.

Base code: [muzairkhattak/multimodal-prompt-learning](https://github.com/muzairkhattak/multimodal-prompt-learning)
at commit [`69bce21`](https://github.com/muzairkhattak/multimodal-prompt-learning/tree/69bce21ae8eda80ad6187534b2dce09cf6c59e17),
with [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) at commit
[`c61a1b5`](https://github.com/KaiyangZhou/Dassl.pytorch/tree/c61a1b570ac6333bd50fb5ae06aea59002fb20bb).
The original README is at [docs/ORIGINAL_README.md](docs/ORIGINAL_README.md). See
[CONTRIBUTIONS.md](CONTRIBUTIONS.md) for what is new in this repository.

## Setup on GAIVI

Run every command from the repository root.

**1. Clone**

```bash
git clone https://github.com/HeyManan/multimodal-prompt-learning.git
cd multimodal-prompt-learning
```

**2. Create the environment**

```bash
conda create -y -n maple python=3.8.20 pip=24.2
conda activate maple
pip install -r requirements.txt

git clone https://github.com/KaiyangZhou/Dassl.pytorch.git ../Dassl.pytorch
git -C ../Dassl.pytorch checkout c61a1b570ac6333bd50fb5ae06aea59002fb20bb
(cd ../Dassl.pytorch && python setup.py develop)
```

**3. Set `$DATA` and `$OUTPUT`**

`$DATA` holds the datasets and `$OUTPUT` holds the run outputs. Each GAIVI storage area has a 100 GB
quota per user, and the four datasets below need about 25 GB. `/general` is GAIVI's hot data area.

```bash
export DATA=/general/$USER/data
export OUTPUT=/general/$USER/output
mkdir -p "$DATA" "$OUTPUT"
```

**4. Download the CLIP weights**

This caches ViT-B/16 in `~/.cache/clip`. Run it on the login node, because the GAIVI docs do not say
whether compute nodes can reach the internet.

```bash
python -c "from clip import clip; clip._download(clip._MODELS['ViT-B/16'])"
```

**5. Check the environment**

```bash
sbatch slurm/test_env.sbatch
cat test_env_<job id>.out
```

The job uses the `ClsGenAI` partition, so change that line on another cluster. It passes when the
last line of the output is `INFO: Environment check passed.`

**6. Download the datasets**

Both scripts read `$DATA`, resume after an interruption, and stop with an error if the result is
incomplete. `nohup` keeps them running after you log out.

```bash
nohup bash tools/data/download_domainnet.sh > download_domainnet.log 2>&1 &
nohup bash tools/data/download_eurosat_dtd_aircraft.sh > download_eurosat_dtd_aircraft.log 2>&1 &
```

`download_domainnet.log` ends with a table of image counts per style:

- `on_disk` is larger than `train` + `test`, because the zips also hold the images that the cleaned
  lists removed.
- `missing` must be 0 for every style.

The cleaned-list choice, the classes missing from the painting lists, and the classes with few
training images are recorded in [data/README.md](data/README.md).

## Smoke test

This runs MaPLe on DTD, seed 1, for 1 epoch, and ends with a `results.json`. Slurm stops it after
10 minutes, and it passes when the last line of the output is `Smoke test passed.` It needs `$DATA`
with DTD in it and `$OUTPUT`. Its results go to `$OUTPUT/smoke/`, so they never reach a table.

```bash
sbatch scripts/smoke_test.sh
tail -n 1 smoke_<job id>.out
```

## Reproduce the paper's base-to-novel results

This covers MaPLe, Co-CoOp and zero-shot CLIP on DTD and EuroSAT, with 16 shots and seeds 1-3. Run
every command from the repository root, with `$DATA` and `$OUTPUT` set and `conda activate maple`.

**1. Build the few-shot caches**

Do this once, before submitting. Training jobs never write these files, and they stop with an error
if one is missing, so two parallel jobs cannot write the same file.

```bash
export PYTHONPATH="${PWD}"
python tools/prepare_fewshot_cache.py --dataset dtd
python tools/prepare_fewshot_cache.py --dataset eurosat
```

**2. Submit the runs**

This one command submits every run. At most 2 run at once, and finished runs are skipped, so it is
safe to repeat after a failure. Add `--force` to redo finished runs. Zero-shot CLIP has no training
and no seeds, so it runs once per dataset.

```bash
bash slurm/submit.sh --datasets dtd eurosat --methods MaPLe CoCoOp ZeroshotCLIP --seeds 1 2 3
```

`slurm/submit.sh` refuses to start while a `base2new` job is queued or running, because the 2-job
limit applies to one submission only. Each run writes to
`$OUTPUT/base2new/<dataset>/<method>/seed<s>/`, and its folder is deleted and restarted if it has no
`results.json`. Job logs are in `$OUTPUT/slurm/`.

**3. Make the table**

```bash
python tools/analysis/collect_results.py
python tools/analysis/reproduction_table.py
```

`results/runs.csv` holds one row per run. `results/tables/reproduction.csv` and `reproduction.md`
compare the mean and sample SD over seeds with the paper's Table 3. Cells with fewer than 3 seeds
are marked preliminary.

To add a method, add an entry to `METHODS` in `utils/methods.py`. To add a dataset, add its split
file to `SPLIT_FILES` in `utils/paths.py`.

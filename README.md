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

The DomainNet smoke test uses the `real` style with 5 test images per class. It runs MaPLe, seed 1,
for 1 epoch, then zero-shot CLIP on base and new. It has the same 10 minute limit and last line, and
needs DomainNet in `$DATA`.

```bash
sbatch scripts/smoke_test_domainnet.sh
tail -n 1 smoke_domainnet_<job id>.out
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
file to `SPLIT_FILES` in `utils/paths.py`. A DomainNet dataset goes in `DOMAINNET_DATASETS` instead.

## DomainNet base-to-novel runs

Each style is one dataset: `domainnet_b2n_clipart`, `_infograph`, `_painting`, `_quickdraw`, `_real`
and `_sketch`. All styles share one class split: labels 0-172 are base and 173-344 are new, in the
label order of the official lists. Run every command from the repository root, with `$DATA` and
`$OUTPUT` set, `conda activate maple` and `export PYTHONPATH="${PWD}"`.

**1. Split files**

The files in [splits/domainnet/](splits/domainnet/README.md) are committed, and training only reads
them, so `tools/prepare_fewshot_cache.py` is not used for DomainNet. They hold 16 training images per
base class for seeds 1-3, validation images from outside that draw, and one fixed test subset of up
to 50 images per class that every method and seed shares.

To regenerate them from the lists and check them against the lists, the images and the dataset
classes:

```bash
python tools/data/prepare_domainnet_splits.py
python tools/data/check_domainnet_splits.py
```

**2. Submit the runs**

The base-to-novel methods are listed below. The time limit of each comes from `utils/methods.py`.

| Method | Trainer | Prompt depth | Learning rate | Slurm limit |
| --- | --- | --- | --- | --- |
| `MaPLe` | MaPLe | 9 | 0.0035 | 1 h |
| `MaPLe-12` | MaPLe | 12 | 0.0035 | 1 h |
| `MaPLe-shallow` | MaPLe | 1 | 0.0035 | 1 h |
| `IVL-12` | IVLP | 12 text, 12 vision | 0.0035 | 1 h |
| `IVL-9` | IVLP | 9 text, 9 vision | 0.0035 | 1 h |
| `DLP-12` | IVLP | 12 text, 0 vision | 0.0025 | 1 h |
| `CoCoOp` | CoCoOp | none | 0.002 | 3 h |
| `ZeroshotCLIP` | ZeroshotCLIP | none | none | 1 h |

```bash
bash slurm/submit.sh --datasets domainnet_b2n_real --methods MaPLe CoCoOp ZeroshotCLIP --seeds 1 2 3
```

A Slurm array has one time limit, so a group with different limits becomes one array per limit. Each
array starts when the one before it ends, so at most 2 jobs still run at once.

## Style distance to ImageNet

Each DomainNet style gets a distance to ImageNet. The distance is computed in the feature space of
the frozen CLIP ViT-B/16 image encoder, with the trainers' test transform, and does not use any
run's results. For each sample seed 1-3, it compares 5000 random images from the style's full train
list with 5000 ImageNet validation images, 5 per class. The features are L2-normalized. Two
measures are computed, both as mean ± sample SD over the 3 samples:

- the Frechet distance (FD) between the two sets
- the nearest-neighbor distance: the mean over style images of 1 minus the highest cosine
  similarity to any ImageNet image

The near and far groups follow a fixed rule. The 3 styles with the lowest mean FD are near, and the
3 with the highest are far.

**1. ImageNet validation images**

These come from the Hugging Face copy of ImageNet-1k, as parquet files in
`$DATA/imagenet_hf/data/`. The download uses a separate conda environment, because the Hugging Face
CLI needs a newer Python than the `maple` environment. First accept the dataset license at
https://huggingface.co/datasets/ILSVRC/imagenet-1k, then run:

```bash
conda create -y -n hf python=3.10
conda activate hf
pip install huggingface_hub
hf auth login
hf download ILSVRC/imagenet-1k --repo-type dataset --include "data/validation*" --local-dir $DATA/imagenet_hf
```

**2. Test the FD code**

```bash
PYTHONPATH="${PWD}" pytest tests/test_fd.py
```

**3. Run the distance job**

The smoke run uses the `real` style and sample seed 1, with 1000 images per side. That is more than
the 512 feature dimensions, so both covariances have full rank. It writes only to
`$OUTPUT/smoke/distance/`, recomputes its features every time, and passes when the last line is
`Style distance smoke test passed.`

```bash
sbatch --time=00:10:00 slurm/style_distance.sbatch --smoke
tail -n 1 style_distance_<job id>.out
```

The full job caches its features in `$OUTPUT/distance/features/`, and a resubmitted job reuses
them. It writes `results/distance/fd.csv` and `results/distance/near_far.json`.
`near_far.json` also shows the ranking by nearest-neighbor distance and whether it would give the
same split. That is reported only, and it never changes the groups.

```bash
sbatch slurm/style_distance.sbatch
```

## DomainNet analysis

These scripts read only `results/runs.csv` and `results/distance/`. They run on a CPU, from the
repository root:

```bash
export PYTHONPATH="${PWD}"
python tools/analysis/domainnet_tables.py
python tools/analysis/domainnet_gap_tables.py
python tools/analysis/domainnet_figures.py
```

- `results/tables/domainnet_average.*`: for each method and seed, base and new are averaged over the
  6 styles, and HM is taken from those averages. The table shows mean ± sample SD over seeds.
- `results/tables/domainnet_per_style.*`: base, new and HM for each style, with the styles in order
  of FD.
- `results/tables/domainnet_gaps.*`: the paired HM gaps per style. Each is one seed minus the same
  seed:

  | Gap | Definition |
  | --- | --- |
  | G1 | MaPLe − IVL-12 |
  | G1' | MaPLe-12 − IVL-12 |
  | G1'' | MaPLe − IVL-9 |
  | G2 | MaPLe − CoCoOp |
  | G3 | IVL-12 − CoCoOp |
  | G4 | MaPLe − MaPLe-12 |
  | G5 | IVL-12 − DLP-12 |
  | G6 | DLP-12 − CoCoOp |

  The script stops if G2 ≠ G6 + G5 + G1 for any style and seed.
- `results/tables/domainnet_trends.*`: for each seed, the mean gap over the far styles minus the
  mean gap over the near styles. A trend is supported only if its mean over seeds is above 0 and
  above 2 SD. The Spearman correlation between FD and the mean gap is descriptive only.
- `results/tables/domainnet_trends_nn_secondary.*`: the same trends with the nearest-neighbor
  near/far split from `near_far.json`. This is a secondary check only. The FD grouping is primary.
- `results/figures/domainnet_gap_vs_fd_{G1,G2}.{pdf,png}`: the gap against FD, one point per style.
  Each style has its own marker shape: filled markers are near styles and open markers are far
  styles. Vertical bars are the gap's SD over seeds, and horizontal bars are FD's SD over the image
  samples.

FD is printed to 3 decimals, because FD on unit-norm features is small. The CSVs keep FD at full
precision. Accuracies and gaps are printed to 2 decimals.

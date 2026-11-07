"""Locations of datasets, configs and run outputs."""
import os
from pathlib import Path

from utils.domainnet import STYLES as DOMAINNET_STYLES
from utils.methods import NUM_SHOTS

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_FOLDER = "base2new"
SMOKE_FOLDER = "smoke"

DOMAINNET_SPLITS_DIR = REPO_ROOT / "splits" / "domainnet"
# Their split files are committed, so they need no split file or cache in $DATA.
DOMAINNET_DATASETS = {"domainnet_b2n_{}".format(style): style for style in DOMAINNET_STYLES}
DOMAINNET_DATASETS["domainnet_b2n_real_smoke"] = "real"

# A dataset class makes a new random split when its split file is absent.
SPLIT_FILES = {
    "dtd": "split_zhou_DescribableTextures.json",
    "eurosat": "split_zhou_EuroSAT.json",
}


def require_env_dir(name):
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError("Expected ${} to name a folder, found it unset or empty.".format(name))
    path = Path(value)
    if not path.is_dir():
        raise RuntimeError("Expected ${} to be an existing folder, found '{}'.".format(name, value))
    return path


def dataset_config(dataset):
    path = REPO_ROOT / "configs" / "datasets" / "{}.yaml".format(dataset)
    if not path.is_file():
        raise FileNotFoundError("Expected a dataset config at {}, found none.".format(path))
    return path


def require_committed_file(path):
    if not path.is_file():
        raise FileNotFoundError(
            "Expected the committed split file {}, found none. It comes from "
            "tools/data/prepare_domainnet_splits.py.".format(path)
        )


def require_split_file(data_root, dataset):
    if dataset in DOMAINNET_DATASETS:
        require_committed_file(domainnet_test_path(DOMAINNET_DATASETS[dataset]))
        return
    if dataset not in SPLIT_FILES:
        raise ValueError("Expected a dataset in {}, found '{}'.".format(sorted(SPLIT_FILES), dataset))
    path = data_root / dataset / SPLIT_FILES[dataset]
    if not path.is_file():
        raise FileNotFoundError(
            "Expected the fixed split at {}, found none. Run the dataset's download "
            "script, because the dataset class would otherwise make a random split.".format(path)
        )


def fewshot_cache_path(data_root, dataset, seed):
    return data_root / dataset / "split_fewshot" / "shot_{}-seed_{}.pkl".format(NUM_SHOTS, seed)


def require_fewshot_cache(data_root, dataset, seed):
    if dataset in DOMAINNET_DATASETS:
        require_committed_file(domainnet_shot_path(DOMAINNET_DATASETS[dataset], seed))
        return
    path = fewshot_cache_path(data_root, dataset, seed)
    if not path.is_file():
        raise FileNotFoundError(
            "Expected the few-shot cache at {}, found none. Build it first with "
            "tools/prepare_fewshot_cache.py, because parallel jobs must not write it.".format(path)
        )


def domainnet_classnames_path():
    return DOMAINNET_SPLITS_DIR / "classnames.json"


def domainnet_shot_path(style, seed):
    return DOMAINNET_SPLITS_DIR / style / "shot_{}-seed_{}.json".format(NUM_SHOTS, seed)


def domainnet_test_path(style):
    return DOMAINNET_SPLITS_DIR / style / "test_subset.json"


def run_dir(output_root, dataset, method_name, seed, smoke=False):
    # Smoke runs have their own folder so collect_results.py never finds them.
    root = output_root / SMOKE_FOLDER if smoke else output_root
    return root / RUNS_FOLDER / dataset / method_name / "seed{}".format(seed)

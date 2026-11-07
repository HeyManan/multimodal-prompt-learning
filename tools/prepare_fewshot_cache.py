"""Build the repo's few-shot cache for one dataset, in one process.

Run this before submitting parallel jobs, so two jobs never write the same cache file:

    PYTHONPATH="${PWD}" python tools/prepare_fewshot_cache.py --dataset dtd
"""
import argparse
import logging
import pickle

from dassl.config import get_cfg_default
from dassl.data.datasets import build_dataset
from dassl.utils import set_random_seed

from train import extend_cfg  # importing train.py also registers the dataset classes
from utils.logging_setup import setup_logging
from utils.methods import DEFAULT_SEEDS, NUM_SHOTS
from utils.paths import dataset_config, fewshot_cache_path, require_env_dir, require_split_file

VAL_SHOTS = min(NUM_SHOTS, 4)  # the dataset classes cap validation at 4 images per class


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True, help="name of a file in configs/datasets")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    return parser.parse_args()


def build_cfg(data_root, dataset, seed):
    cfg = get_cfg_default()
    extend_cfg(cfg)
    cfg.merge_from_file(str(dataset_config(dataset)))
    cfg.DATASET.ROOT = str(data_root)
    cfg.DATASET.NUM_SHOTS = NUM_SHOTS
    cfg.SEED = seed
    cfg.freeze()
    return cfg


def check_cache(cache_path, num_classes):
    with cache_path.open("rb") as file:
        data = pickle.load(file)
    expected_train = NUM_SHOTS * num_classes
    expected_val = VAL_SHOTS * num_classes
    found_train, found_val = len(data["train"]), len(data["val"])
    if (found_train, found_val) != (expected_train, expected_val):
        raise RuntimeError(
            "Expected {} train and {} val images in {}, found {} and {}.".format(
                expected_train, expected_val, cache_path, found_train, found_val
            )
        )
    logging.info("%s: %d train, %d val images", cache_path, found_train, found_val)


def prepare_seed(data_root, dataset, seed):
    cache_path = fewshot_cache_path(data_root, dataset, seed)
    if cache_path.exists():
        logging.info("Checking the existing cache %s", cache_path)
    else:
        logging.info("Building %s", cache_path)
    # Same order as train.py, so the draw matches what a job for this seed would make.
    set_random_seed(seed)
    dataset_object = build_dataset(build_cfg(data_root, dataset, seed))
    check_cache(cache_path, dataset_object.num_classes)


def main():
    setup_logging()
    args = parse_args()
    data_root = require_env_dir("DATA")
    require_split_file(data_root, args.dataset)
    for seed in args.seeds:
        if seed < 1:
            raise ValueError("Expected seeds >= 1, because train.py ignores seed 0, found {}.".format(seed))
        prepare_seed(data_root, args.dataset, seed)
    logging.info("Few-shot cache ready for %s, seeds %s.", args.dataset, args.seeds)


if __name__ == "__main__":
    main()

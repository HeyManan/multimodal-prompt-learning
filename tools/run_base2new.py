"""Run one (method, dataset, seed) in the base-to-novel setting and write results.json.

A trained method runs train on base (which ends with the test on base), then --eval-only on new.
A method without training runs --eval-only on base and on new. Called by slurm/run_one.sbatch.
"""
import argparse
import logging
import shutil
import subprocess
import sys
import time

import torch
import yaml

from utils.log_parsing import parse_counts
from utils.logging_setup import setup_logging
from utils.methods import NO_SEED, NUM_SHOTS, SMOKE_EPOCHS, ZEROSHOT_RNG_SEED, get_method
from utils.paths import (
    REPO_ROOT,
    dataset_config,
    require_env_dir,
    require_fewshot_cache,
    require_split_file,
    run_dir,
)
from utils.run_record import accuracy, git_commit, harmonic_mean, slurm_job_id, write_results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--method", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, help="required for trained methods, >= 1")
    parser.add_argument("--force", action="store_true", help="redo a finished run")
    parser.add_argument("--smoke", action="store_true", help="1 epoch if trained, results in $OUTPUT/smoke")
    return parser.parse_args()


def check_seed(method, seed):
    if method.is_trained and (seed is None or seed < 1):
        raise ValueError(
            "Expected --seed >= 1 for a trained method, because train.py ignores "
            "seed 0, found {}.".format(seed)
        )
    if not method.is_trained and seed is not None:
        raise ValueError("Expected no --seed for {}, found {}.".format(method.trainer, seed))


def require_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("Expected a usable GPU, found torch.cuda.is_available() == False.")


def read_settings(method, smoke):
    """Return (epochs, lr, batch size) as recorded in results.json."""
    if not method.is_trained:
        return 0, None, None
    config = yaml.safe_load((REPO_ROOT / method.config_file).read_text())
    max_epoch = config["OPTIM"]["MAX_EPOCH"]
    if max_epoch != method.final_epoch:
        raise RuntimeError(
            "Expected MAX_EPOCH {} in {}, found {}.".format(
                method.final_epoch, method.config_file, max_epoch
            )
        )
    epochs = SMOKE_EPOCHS if smoke else max_epoch
    return epochs, config["OPTIM"]["LR"], config["DATALOADER"]["TRAIN_X"]["BATCH_SIZE"]


def train_py_command(method, dataset, rng_seed, data_root, output_dir):
    return [
        sys.executable, "-u", "train.py",
        "--root", str(data_root),
        "--seed", str(rng_seed),
        "--trainer", method.trainer,
        "--dataset-config-file", str(dataset_config(dataset)),
        "--config-file", method.config_file,
        "--output-dir", str(output_dir),
    ]


def config_options(method, subsample, smoke):
    options = ["DATASET.SUBSAMPLE_CLASSES", subsample]
    # Zero-shot must not set NUM_SHOTS: it would write a cache file, and the test
    # images and the base/new split do not depend on it.
    if method.is_trained:
        options += ["DATASET.NUM_SHOTS", str(NUM_SHOTS)]
    if smoke:
        options += ["OPTIM.MAX_EPOCH", str(SMOKE_EPOCHS)]
    return options


def run_timed(command):
    logging.info("Running: %s", " ".join(command))
    start = time.monotonic()
    subprocess.run(command, cwd=str(REPO_ROOT), check=True)
    return time.monotonic() - start


def run_trained_method(method, dataset, seed, epochs, data_root, run_root, smoke):
    train_dir = run_root / "train_base"
    new_dir = run_root / "test_new"
    train_seconds = run_timed(
        train_py_command(method, dataset, seed, data_root, train_dir)
        + config_options(method, "base", smoke)
    )
    eval_seconds = run_timed(
        train_py_command(method, dataset, seed, data_root, new_dir)
        + ["--model-dir", str(train_dir), "--load-epoch", str(epochs), "--eval-only"]
        + config_options(method, "new", smoke)
    )
    return train_dir, new_dir, train_seconds, eval_seconds


def run_untrained_method(method, dataset, data_root, run_root):
    base_dir = run_root / "test_base"
    new_dir = run_root / "test_new"
    eval_seconds = 0.0
    for output_dir, subsample in ((base_dir, "base"), (new_dir, "new")):
        eval_seconds += run_timed(
            train_py_command(method, dataset, ZEROSHOT_RNG_SEED, data_root, output_dir)
            + ["--eval-only"]
            + config_options(method, subsample, smoke=False)
        )
    return base_dir, new_dir, None, eval_seconds


def build_record(args, method, rng_seed, settings, dirs, seconds):
    epochs, lr, batch_size = settings
    base_correct, base_total = parse_counts(dirs[0] / "log.txt")
    new_correct, new_total = parse_counts(dirs[1] / "log.txt")
    base_acc = accuracy(base_correct, base_total)
    new_acc = accuracy(new_correct, new_total)
    return {
        "method": args.method,
        "dataset": args.dataset,
        "seed": args.seed if method.is_trained else NO_SEED,
        "rng_seed": rng_seed,
        "base_acc": base_acc,
        "new_acc": new_acc,
        "hm": harmonic_mean(base_acc, new_acc),
        "base_correct": base_correct,
        "base_total": base_total,
        "new_correct": new_correct,
        "new_total": new_total,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "num_shots": NUM_SHOTS if method.is_trained else None,
        "config_file": method.config_file,
        "dataset_config_file": str(dataset_config(args.dataset).relative_to(REPO_ROOT)),
        "git_commit": git_commit(),
        "gpu_name": torch.cuda.get_device_name(0),
        "slurm_job_id": slurm_job_id(),
        "train_seconds": seconds[0],
        "eval_seconds": seconds[1],
        "smoke": args.smoke,
    }


def main():
    setup_logging()
    args = parse_args()
    method = get_method(args.method)
    check_seed(method, args.seed)

    data_root = require_env_dir("DATA")
    output_root = require_env_dir("OUTPUT")
    seed_label = args.seed if method.is_trained else NO_SEED
    run_root = run_dir(output_root, args.dataset, args.method, seed_label, args.smoke)
    results_path = run_root / "results.json"

    if results_path.exists() and not args.force:
        logging.info("Skipping finished run: %s", results_path)
        return

    require_split_file(data_root, args.dataset)
    if method.is_trained:
        require_fewshot_cache(data_root, args.dataset, args.seed)
    require_gpu()
    settings = read_settings(method, args.smoke)

    # Dassl resumes from any checkpoint in the output folder and keeps old logs,
    # so a leftover folder could mix two runs.
    if run_root.exists():
        reason = "--force was given" if args.force else "it has no results.json, so the run did not finish"
        logging.warning("Deleting run folder %s because %s.", run_root, reason)
        shutil.rmtree(str(run_root))

    if method.is_trained:
        rng_seed = args.seed
        base_dir, new_dir, train_seconds, eval_seconds = run_trained_method(
            method, args.dataset, args.seed, settings[0], data_root, run_root, args.smoke
        )
    else:
        rng_seed = ZEROSHOT_RNG_SEED
        base_dir, new_dir, train_seconds, eval_seconds = run_untrained_method(
            method, args.dataset, data_root, run_root
        )

    record = build_record(
        args, method, rng_seed, settings, (base_dir, new_dir), (train_seconds, eval_seconds)
    )
    write_results(results_path, record)
    logging.info(
        "%s %s seed %s: base %.2f, new %.2f, HM %.2f. Wrote %s",
        args.method, args.dataset, record["seed"], record["base_acc"], record["new_acc"],
        record["hm"], results_path,
    )


if __name__ == "__main__":
    main()

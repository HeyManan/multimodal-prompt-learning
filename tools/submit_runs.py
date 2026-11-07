"""Submit a group of runs as one Slurm array, skipping finished runs. Called by slurm/submit.sh."""
import argparse
import getpass
import logging
import subprocess
import time

from utils.logging_setup import setup_logging
from utils.methods import NO_SEED, get_method
from utils.paths import (
    REPO_ROOT,
    require_env_dir,
    require_fewshot_cache,
    require_split_file,
    run_dir,
)

JOB_NAME = "base2new"
MAX_CONCURRENT_JOBS = 2
SLURM_FOLDER = "slurm"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[], help="for trained methods")
    parser.add_argument("--force", action="store_true", help="redo finished runs")
    return parser.parse_args()


def require_no_active_jobs():
    # %2 limits one array only. A second array could pass the limit, and it could
    # hold a run that is still pending in the first array, so one group runs at a time.
    command = ["squeue", "-h", "-u", getpass.getuser(), "-n", JOB_NAME, "-o", "%i"]
    active = subprocess.check_output(command, universal_newlines=True).split()
    if active:
        raise RuntimeError(
            "Expected no active {} jobs, found {}: {}. Wait for them to finish.".format(
                JOB_NAME, len(active), " ".join(active)
            )
        )


def expand_runs(datasets, method_names, seeds):
    """Return (dataset, method name, seed) tuples. Methods without training run once."""
    runs = []
    for dataset in datasets:
        for method_name in method_names:
            method = get_method(method_name)
            if not method.is_trained:
                runs.append((dataset, method_name, None))
                continue
            if not seeds or min(seeds) < 1:
                raise ValueError(
                    "Expected --seeds >= 1 for {}, because train.py ignores seed 0, "
                    "found {}.".format(method_name, seeds)
                )
            runs += [(dataset, method_name, seed) for seed in seeds]
    return runs


def is_finished(output_root, run):
    dataset, method_name, seed = run
    seed_label = NO_SEED if seed is None else seed
    return (run_dir(output_root, dataset, method_name, seed_label) / "results.json").exists()


def check_inputs(data_root, runs):
    for dataset, _, seed in runs:
        require_split_file(data_root, dataset)
        if seed is not None:
            require_fewshot_cache(data_root, dataset, seed)


def task_line(run, force):
    dataset, method_name, seed = run
    line = "--method {} --dataset {}".format(method_name, dataset)
    if seed is not None:
        line += " --seed {}".format(seed)
    if force:
        line += " --force"
    return line


def write_task_file(slurm_dir, runs, force):
    path = slurm_dir / "tasks_{}.txt".format(time.strftime("%Y%m%d-%H%M%S"))
    path.write_text("".join(task_line(run, force) + "\n" for run in runs))
    return path


def submit_array(slurm_dir, task_file, num_tasks):
    command = [
        "sbatch",
        "--array=1-{}%{}".format(num_tasks, MAX_CONCURRENT_JOBS),
        "--job-name={}".format(JOB_NAME),
        "--output={}".format(slurm_dir / "%x_%A_%a.out"),
        "slurm/run_one.sbatch",
        str(task_file),
    ]
    logging.info("Running: %s", " ".join(command))
    subprocess.run(command, cwd=str(REPO_ROOT), check=True)


def main():
    setup_logging()
    args = parse_args()
    data_root = require_env_dir("DATA")
    output_root = require_env_dir("OUTPUT")
    require_no_active_jobs()

    runs = expand_runs(args.datasets, args.methods, args.seeds)
    pending = [run for run in runs if args.force or not is_finished(output_root, run)]
    check_inputs(data_root, pending)

    if pending:
        slurm_dir = output_root / SLURM_FOLDER
        slurm_dir.mkdir(exist_ok=True)
        task_file = write_task_file(slurm_dir, pending, args.force)
        submit_array(slurm_dir, task_file, len(pending))
    logging.info("Submitted %d runs, skipped %d finished runs.", len(pending), len(runs) - len(pending))


if __name__ == "__main__":
    main()

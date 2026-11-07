"""Helpers for the results.json file of a run."""
import json
import os
import subprocess

from utils.paths import REPO_ROOT


def accuracy(correct, total):
    return 100.0 * correct / total


def harmonic_mean(base_acc, new_acc):
    return 2 * base_acc * new_acc / (base_acc + new_acc)


def git_commit():
    command = ["git", "rev-parse", "HEAD"]
    return subprocess.check_output(command, cwd=str(REPO_ROOT), universal_newlines=True).strip()


def slurm_job_id():
    return os.environ.get("SLURM_JOB_ID", "none")


def write_results(path, record):
    # results.json marks a run as finished, so it must never exist half-written.
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(record, indent=2) + "\n")
    temporary_path.replace(path)

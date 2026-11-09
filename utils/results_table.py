"""Reading results/runs.csv and formatting numbers, shared by the table scripts."""
import csv
from statistics import stdev

from utils.paths import REPO_ROOT

RUNS_CSV = REPO_ROOT / "results" / "runs.csv"
TABLES_DIR = REPO_ROOT / "results" / "tables"
MIN_SEEDS = 3


def read_runs(csv_path):
    if not csv_path.is_file():
        raise FileNotFoundError(
            "Expected {}, found none. Run tools/analysis/collect_results.py first.".format(csv_path)
        )
    with csv_path.open(newline="") as file:
        return list(csv.DictReader(file))


def check_unique_runs(runs):
    keys = [(run["dataset"], run["method"], run["seed"]) for run in runs]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Expected one row per dataset, method and seed, found duplicates.")


def check_same_test_images(runs):
    """Every method on a dataset must be tested on the same number of images."""
    for dataset in {run["dataset"] for run in runs}:
        totals = {(run["base_total"], run["new_total"]) for run in runs if run["dataset"] == dataset}
        if len(totals) != 1:
            raise RuntimeError(
                "Expected one (base_total, new_total) for {}, found {}.".format(dataset, sorted(totals))
            )


def sample_sd(values):
    # SD is undefined for one seed, and a run without seeds has no spread at all.
    return stdev(values) if len(values) >= 2 else None


def format_number(value, decimals=2):
    return "" if value is None else "{:.{}f}".format(value, decimals)


def mean_sd_cell(mean_text, sd_text, preliminary=False):
    """'mean ± SD' for a Markdown cell; a run without seeds shows its value alone."""
    cell = mean_text + (" ± " + sd_text if sd_text else "")
    return cell + (" (prelim.)" if preliminary else "")


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, header, rows, notes, title=None):
    lines = ["**{}**".format(title), ""] if title else []
    lines += ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    lines += ["| " + " | ".join(cells) + " |" for cells in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines + [""] + notes) + "\n")

"""DomainNet accuracy tables: the average over the 6 styles, and each style ordered by FD to ImageNet.

Reads only results/runs.csv and results/distance/. HM is computed per seed and then averaged:
for the average table, each seed's HM comes from that seed's base and new averaged over the styles.
Writes domainnet_average.{csv,md} and domainnet_per_style.{csv,md} to results/tables/:

    PYTHONPATH="${PWD}" python tools/analysis/domainnet_tables.py
"""
import logging
from statistics import mean

from utils.domainnet import STYLES
from utils.domainnet_gaps import domainnet_runs, method_seeds, run_hm
from utils.logging_setup import setup_logging
from utils.methods import METHODS
from utils.results_table import (
    MIN_SEEDS,
    RUNS_CSV,
    TABLES_DIR,
    format_number,
    mean_sd_cell,
    read_runs,
    sample_sd,
    write_csv,
    write_markdown,
)
from utils.run_record import harmonic_mean
from utils.style_distance import read_style_distances

METRICS = ("base", "new", "hm")
FD_DECIMALS = 3


def summary(method_name, values_by_metric):
    """n_seeds, preliminary flag and mean and SD over seeds of base, new and HM."""
    trained = METHODS[method_name].is_trained
    n_seeds = len(values_by_metric["hm"])
    row = {
        "method": method_name,
        "n_seeds": n_seeds if trained else "n/a",
        "preliminary": trained and n_seeds < MIN_SEEDS,
    }
    for metric in METRICS:
        row[metric + "_mean"] = format_number(mean(values_by_metric[metric]))
        row[metric + "_sd"] = format_number(sample_sd(values_by_metric[metric]))
    return row


def average_rows(runs):
    rows = []
    for method_name in METHODS:
        seeds = method_seeds(method_name)
        base = [mean(float(runs[(method_name, style, seed)]["base_acc"]) for style in STYLES) for seed in seeds]
        new = [mean(float(runs[(method_name, style, seed)]["new_acc"]) for style in STYLES) for seed in seeds]
        hm = [harmonic_mean(b, n) for b, n in zip(base, new)]
        rows.append(summary(method_name, {"base": base, "new": new, "hm": hm}))
    return rows


def per_style_rows(runs, distances):
    rows = []
    for distance in distances:
        for method_name in METHODS:
            selected = [runs[(method_name, distance.style, seed)] for seed in method_seeds(method_name)]
            values = {
                "base": [float(run["base_acc"]) for run in selected],
                "new": [float(run["new_acc"]) for run in selected],
                "hm": [run_hm(run) for run in selected],
            }
            row = {
                "style": distance.style,
                "group": distance.group,
                # Full precision in the CSV; the Markdown table rounds it.
                "fd_mean": distance.fd_mean,
                "fd_sd": distance.fd_sd,
            }
            row.update(summary(method_name, values))
            rows.append(row)
    return rows


def metric_cells(row):
    return [mean_sd_cell(row[metric + "_mean"], row[metric + "_sd"], row["preliminary"]) for metric in METRICS]


NOTES = [
    "Accuracy in %, mean ± sample SD over seeds. HM is computed per seed, then averaged.",
    "Zero-shot CLIP has no training and one run, so it has no SD. "
    "(prelim.) marks fewer than {} seeds.".format(MIN_SEEDS),
]


def write_average(rows):
    cells = [[row["method"], str(row["n_seeds"])] + metric_cells(row) for row in rows]
    write_markdown(
        TABLES_DIR / "domainnet_average.md",
        ["Method", "Seeds", "Base", "New", "HM"],
        cells,
        NOTES + ["Per seed, base and new are averaged over the 6 styles, and HM is taken from those averages."],
    )
    write_csv(rows, TABLES_DIR / "domainnet_average.csv")


def write_per_style(rows):
    cells = []
    for row in rows:
        style = "{} ({})".format(row["style"], row["group"])
        fd = mean_sd_cell(format_number(row["fd_mean"], FD_DECIMALS), format_number(row["fd_sd"], FD_DECIMALS))
        cells.append([style, fd, row["method"], str(row["n_seeds"])] + metric_cells(row))
    write_markdown(
        TABLES_DIR / "domainnet_per_style.md",
        ["Style (group)", "FD to ImageNet", "Method", "Seeds", "Base", "New", "HM"],
        cells,
        NOTES + ["Styles in order of mean FD to ImageNet: mean ± sample SD over the image samples, 3 decimals."],
    )
    write_csv(rows, TABLES_DIR / "domainnet_per_style.csv")


def main():
    setup_logging()
    runs = domainnet_runs(read_runs(RUNS_CSV))
    distances = read_style_distances()
    write_average(average_rows(runs))
    write_per_style(per_style_rows(runs, distances))
    logging.info("Wrote the DomainNet average and per-style tables to %s", TABLES_DIR)


if __name__ == "__main__":
    main()

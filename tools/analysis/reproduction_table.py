"""Compare results/runs.csv with the paper: base, new and HM per dataset and method.

Mean HM is computed from the averaged base and new accuracies, as the paper does. The SD of HM is
the sample SD of the per-seed HMs. Writes CSV and Markdown to results/tables/.
"""
import csv
import logging
from statistics import mean

from utils.logging_setup import setup_logging
from utils.methods import NO_SEED
from utils.paper_results import PAPER_RESULTS
from utils.results_table import (
    MIN_SEEDS,
    RUNS_CSV,
    TABLES_DIR,
    check_same_test_images,
    check_unique_runs,
    format_number,
    read_runs,
    sample_sd,
)
from utils.run_record import harmonic_mean

METRICS = ("base", "new", "hm")
CSV_FIELDS = ["dataset", "method", "n_seeds", "preliminary"] + [
    "{}_{}".format(metric, part) for metric in METRICS for part in ("mean", "sd", "paper", "diff")
]


def summarize(dataset, method, runs, paper):
    base = [float(run["base_acc"]) for run in runs]
    new = [float(run["new_acc"]) for run in runs]
    per_seed_hm = [harmonic_mean(b, n) for b, n in zip(base, new)]
    means = (mean(base), mean(new), harmonic_mean(mean(base), mean(new)))
    sds = (sample_sd(base), sample_sd(new), sample_sd(per_seed_hm))

    has_seeds = runs[0]["seed"] != NO_SEED
    row = {
        "dataset": dataset,
        "method": method,
        "n_seeds": len(runs) if has_seeds else "n/a",
        "preliminary": has_seeds and len(runs) < MIN_SEEDS,
    }
    for metric, metric_mean, sd, paper_value in zip(METRICS, means, sds, paper):
        row[metric + "_mean"] = format_number(metric_mean)
        row[metric + "_sd"] = format_number(sd)
        row[metric + "_paper"] = format_number(paper_value)
        # Adding 0.0 turns a rounded -0.0 into 0.0, so it does not print as -0.00.
        row[metric + "_diff"] = "{:+.2f}".format(round(metric_mean - paper_value, 2) + 0.0)
    return row


def build_rows(runs):
    rows = []
    for (dataset, method), paper in PAPER_RESULTS.items():
        selected = [run for run in runs if run["dataset"] == dataset and run["method"] == method]
        if not selected:
            logging.warning("No runs for %s on %s, so it is left out of the table.", method, dataset)
            continue
        rows.append(summarize(dataset, method, selected, paper))
    if not rows:
        raise RuntimeError("Expected runs for at least one paper target, found none.")
    return rows


def write_csv(rows, path):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def format_cell(row, metric):
    cell = row[metric + "_mean"]
    if row["n_seeds"] != "n/a":
        cell += " ± " + (row[metric + "_sd"] or "n/a")
    if row["preliminary"]:
        cell += " (prelim.)"
    return cell


def write_markdown(rows, path):
    header = ["Dataset", "Method", "Seeds"]
    for metric in ("Base", "New", "HM"):
        header += [metric, metric + " paper", metric + " diff"]
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    for row in rows:
        cells = [row["dataset"], row["method"], str(row["n_seeds"])]
        for metric in METRICS:
            cells += [format_cell(row, metric), row[metric + "_paper"], row[metric + "_diff"]]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "Accuracy in %, mean ± sample SD over seeds. Diff is ours minus the paper's Table 3.",
        "Mean HM is computed from the averaged base and new accuracies.",
        "(prelim.) marks fewer than {} seeds. Seeds n/a marks a run with no training, "
        "which has no seeds.".format(MIN_SEEDS),
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    setup_logging()
    runs = read_runs(RUNS_CSV)
    check_unique_runs(runs)
    check_same_test_images(runs)
    rows = build_rows(runs)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(rows, TABLES_DIR / "reproduction.csv")
    write_markdown(rows, TABLES_DIR / "reproduction.md")
    logging.info("Wrote %d rows to %s", len(rows), TABLES_DIR)


if __name__ == "__main__":
    main()

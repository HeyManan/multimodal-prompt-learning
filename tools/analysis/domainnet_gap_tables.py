"""Paired HM gaps between methods on each DomainNet style, and their trend from near to far styles.

Reads only results/runs.csv and results/distance/. A gap is one seed's HM minus the same seed's HM
of another method. A seed's trend is the mean gap over the far styles minus the mean gap over the
near styles. A trend is "supported" only if its mean over seeds is above 0 and above 2 SD. The
Spearman correlation between FD and mean gap over the 6 styles is descriptive only.
Writes domainnet_gaps.{csv,md} and domainnet_trends.{csv,md} to results/tables/:

    PYTHONPATH="${PWD}" python tools/analysis/domainnet_gap_tables.py
"""
import logging
from statistics import mean

from scipy.stats import spearmanr

from utils.domainnet_gaps import GAPS, SEEDS, check_gap_identity, domainnet_runs, gap_by_seed
from utils.logging_setup import setup_logging
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
from utils.style_distance import FAR, NEAR, read_style_distances

FD_DECIMALS = 3
SUPPORT_SDS = 2


def gap_rows(runs, distances):
    rows = []
    for gap in GAPS:
        for distance in distances:
            values = gap_by_seed(runs, gap, distance.style)
            row = {
                "gap": gap.name,
                "definition": gap.definition,
                "style": distance.style,
                "group": distance.group,
                # Full precision in the CSV; the Markdown table rounds it.
                "fd_mean": distance.fd_mean,
            }
            for seed in SEEDS:
                row["gap_seed" + seed] = format_number(values[seed])
            row["gap_mean"] = format_number(mean(values.values()))
            row["gap_sd"] = format_number(sample_sd(list(values.values())))
            row["n_seeds"] = len(SEEDS)
            rows.append(row)
    return rows


def trend_by_seed(runs, gap, distances):
    """{seed: mean gap over far styles minus mean gap over near styles}."""
    values = {distance.style: gap_by_seed(runs, gap, distance.style) for distance in distances}
    near = [distance.style for distance in distances if distance.group == NEAR]
    far = [distance.style for distance in distances if distance.group == FAR]
    return {
        seed: mean(values[style][seed] for style in far) - mean(values[style][seed] for style in near)
        for seed in SEEDS
    }


def spearman_fd_gap(runs, gap, distances):
    mean_gaps = [mean(gap_by_seed(runs, gap, distance.style).values()) for distance in distances]
    # Descriptive only: with 6 styles the p-value says little, so it is not reported.
    rho, _ = spearmanr([distance.fd_mean for distance in distances], mean_gaps)
    return rho


def trend_rows(runs, distances):
    rows = []
    for gap in GAPS:
        trends = trend_by_seed(runs, gap, distances)
        trend_mean = mean(trends.values())
        trend_sd = sample_sd(list(trends.values()))
        row = {"gap": gap.name, "definition": gap.definition}
        for seed in SEEDS:
            row["trend_seed" + seed] = format_number(trends[seed])
        row["trend_mean"] = format_number(trend_mean)
        row["trend_sd"] = format_number(trend_sd)
        row["n_seeds"] = len(SEEDS)
        row["preliminary"] = len(SEEDS) < MIN_SEEDS
        row["supported"] = trend_mean > 0 and trend_mean > SUPPORT_SDS * trend_sd
        row["spearman_fd_vs_mean_gap"] = format_number(spearman_fd_gap(runs, gap, distances))
        rows.append(row)
    return rows


def write_gaps(rows, distances):
    header = ["Gap", "Definition"] + [
        "{} ({}, FD {})".format(distance.style, distance.group, format_number(distance.fd_mean, FD_DECIMALS))
        for distance in distances
    ]
    cells = []
    for gap in GAPS:
        by_style = {row["style"]: row for row in rows if row["gap"] == gap.name}
        style_cells = [
            mean_sd_cell(by_style[distance.style]["gap_mean"], by_style[distance.style]["gap_sd"])
            for distance in distances
        ]
        cells.append([gap.name, gap.definition] + style_cells)
    write_markdown(
        TABLES_DIR / "domainnet_gaps.md",
        header,
        cells,
        [
            "HM gap in percentage points, mean ± sample SD over {} seeds. Each value is one seed's HM "
            "minus the same seed's HM of the other method.".format(len(SEEDS)),
            "Styles in order of mean FD to ImageNet. G2 = G6 + G5 + G1 was checked for every style and seed.",
        ],
    )
    write_csv(rows, TABLES_DIR / "domainnet_gaps.csv")


def write_trends(rows):
    cells = [
        [
            row["gap"],
            row["definition"],
            mean_sd_cell(row["trend_mean"], row["trend_sd"], row["preliminary"]),
            "yes" if row["supported"] else "no",
            row["spearman_fd_vs_mean_gap"],
        ]
        for row in rows
    ]
    write_markdown(
        TABLES_DIR / "domainnet_trends.md",
        ["Gap", "Definition", "Far − near", "Supported", "Spearman ρ (FD, mean gap)"],
        cells,
        [
            "Far − near: per seed, the mean gap over the 3 far styles minus the mean gap over the 3 near "
            "styles, in percentage points; mean ± sample SD over {} seeds.".format(len(SEEDS)),
            "Supported only if the mean is above 0 and above {} SD.".format(SUPPORT_SDS),
            "Spearman ρ over the 6 styles between mean FD and mean gap is descriptive only; no p-value.",
        ],
    )
    write_csv(rows, TABLES_DIR / "domainnet_trends.csv")


def main():
    setup_logging()
    runs = domainnet_runs(read_runs(RUNS_CSV))
    check_gap_identity(runs)
    distances = read_style_distances()
    write_gaps(gap_rows(runs, distances), distances)
    write_trends(trend_rows(runs, distances))
    logging.info("Wrote the DomainNet gap and trend tables to %s", TABLES_DIR)


if __name__ == "__main__":
    main()

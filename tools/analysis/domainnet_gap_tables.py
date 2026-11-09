"""Paired HM gaps between methods on each DomainNet style, and their trend from near to far styles.

Reads only results/runs.csv and results/distance/. A gap is one seed's HM minus the same seed's HM
of another method. A seed's trend is the mean gap over the far styles minus the mean gap over the
near styles. A trend is "supported" only if its mean over seeds is above 0 and above 2 SD. The
Spearman correlation between a style distance and the mean gap is descriptive only.
The FD near/far groups are primary. Two secondary trend tables check them: one uses the
nearest-neighbor groups and NN distance, the other leaves quickdraw out.
Writes domainnet_gaps, domainnet_trends, domainnet_trends_nn_secondary and
domainnet_trends_leave_quickdraw_out_secondary (CSV and Markdown) to results/tables/:

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
from utils.style_distance import FAR, NEAR, read_nn_groups, read_style_distances

FD_DECIMALS = 3
SUPPORT_SDS = 2
FD_GROUPING = "primary, FD grouping"
NN_GROUPING = "secondary, NN grouping"
LEAVE_OUT_GROUPING = "secondary, leave-quickdraw-out"
# quickdraw has the largest FD and the largest gaps, and changes group between FD and NN, so this
# check shows whether the FD trend holds without it.
LEFT_OUT_STYLE = "quickdraw"
TREND_HEADER = ["Gap", "Definition", "Far − near", "Supported"]
TREND_NOTES = [
    "Far − near: per seed, the mean gap over the far styles minus the mean gap over the near "
    "styles, in percentage points; mean ± sample SD over {} seeds.".format(len(SEEDS)),
    "Supported only if the mean is above 0 and above {} SD.".format(SUPPORT_SDS),
]


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


def trend_by_seed(runs, gap, groups):
    """{seed: mean gap over far styles minus mean gap over near styles}, for groups {style: near or far}."""
    values = {style: gap_by_seed(runs, gap, style) for style in groups}
    near = [style for style, group in groups.items() if group == NEAR]
    far = [style for style, group in groups.items() if group == FAR]
    return {
        seed: mean(values[style][seed] for style in far) - mean(values[style][seed] for style in near)
        for seed in SEEDS
    }


def spearman_gap(runs, gap, distance_by_style):
    """Spearman rho between a style distance and the mean gap, over the styles in distance_by_style."""
    styles = list(distance_by_style)
    mean_gaps = [mean(gap_by_seed(runs, gap, style).values()) for style in styles]
    # Descriptive only: with 5 or 6 styles the p-value says little, so it is not reported.
    rho, _ = spearmanr([distance_by_style[style] for style in styles], mean_gaps)
    return rho


def trend_row(runs, gap, grouping, groups):
    trends = trend_by_seed(runs, gap, groups)
    trend_mean = mean(trends.values())
    trend_sd = sample_sd(list(trends.values()))
    row = {"grouping": grouping, "gap": gap.name, "definition": gap.definition}
    for seed in SEEDS:
        row["trend_seed" + seed] = format_number(trends[seed])
    row["trend_mean"] = format_number(trend_mean)
    row["trend_sd"] = format_number(trend_sd)
    row["n_seeds"] = len(SEEDS)
    row["preliminary"] = len(SEEDS) < MIN_SEEDS
    row["supported"] = trend_mean > 0 and trend_mean > SUPPORT_SDS * trend_sd
    return row


def trend_rows(runs, grouping, groups, measure, distance_by_style):
    rows = []
    for gap in GAPS:
        row = trend_row(runs, gap, grouping, groups)
        row["spearman_measure"] = measure
        row["spearman_rho"] = format_number(spearman_gap(runs, gap, distance_by_style))
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


def trend_cells(row):
    trend = mean_sd_cell(row["trend_mean"], row["trend_sd"], row["preliminary"])
    return [row["gap"], row["definition"], trend, "yes" if row["supported"] else "no"]


def group_note(groups):
    near = ", ".join(style for style, group in groups.items() if group == NEAR)
    far = ", ".join(style for style, group in groups.items() if group == FAR)
    return "Near: {}. Far: {}.".format(near, far)


def write_trends(rows, file_stem, groups, notes):
    """One trend table; notes say where the groups come from and which table is primary."""
    grouping, measure = rows[0]["grouping"], rows[0]["spearman_measure"]
    write_markdown(
        TABLES_DIR / (file_stem + ".md"),
        TREND_HEADER + ["Spearman ρ ({}, mean gap)".format(measure)],
        [trend_cells(row) + [row["spearman_rho"]] for row in rows],
        [group_note(groups)]
        + notes
        + TREND_NOTES
        + [
            "Spearman ρ over the {} styles above between mean {} and mean gap is descriptive only; "
            "no p-value.".format(len(groups), measure)
        ],
        title="Trend from near to far styles ({})".format(grouping),
    )
    write_csv(rows, TABLES_DIR / (file_stem + ".csv"))


def write_all_trends(runs, distances):
    fd = {distance.style: distance.fd_mean for distance in distances}
    nn = {distance.style: distance.nn_mean for distance in distances}
    fd_groups = {distance.style: distance.group for distance in distances}
    write_trends(trend_rows(runs, FD_GROUPING, fd_groups, "FD", fd), "domainnet_trends", fd_groups, [])

    nn_groups = read_nn_groups()
    write_trends(
        trend_rows(runs, NN_GROUPING, nn_groups, "NN distance", nn),
        "domainnet_trends_nn_secondary",
        nn_groups,
        [
            "Groups from the nearest-neighbor ranking in results/distance/near_far.json, with the same "
            "3 near / 3 far rule. Secondary check only: the FD grouping in domainnet_trends.md is primary."
        ],
    )

    kept_groups = {style: group for style, group in fd_groups.items() if style != LEFT_OUT_STYLE}
    write_trends(
        trend_rows(runs, LEAVE_OUT_GROUPING, kept_groups, "FD", {style: fd[style] for style in kept_groups}),
        "domainnet_trends_leave_quickdraw_out_secondary",
        kept_groups,
        [
            "The FD groups with {} left out. Secondary check only: the FD grouping in "
            "domainnet_trends.md is primary.".format(LEFT_OUT_STYLE)
        ],
    )


def main():
    setup_logging()
    runs = domainnet_runs(read_runs(RUNS_CSV))
    check_gap_identity(runs)
    distances = read_style_distances()
    write_gaps(gap_rows(runs, distances), distances)
    write_all_trends(runs, distances)
    logging.info("Wrote the DomainNet gap and trend tables to %s", TABLES_DIR)


if __name__ == "__main__":
    main()

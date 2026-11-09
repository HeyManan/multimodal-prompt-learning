"""Gap vs FD to ImageNet on DomainNet: one point per style, for G1 and G2.

Reads only results/runs.csv and results/distance/. Writes PDF and PNG to results/figures/:

    PYTHONPATH="${PWD}" python tools/analysis/domainnet_figures.py
"""
import logging
from statistics import mean

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from utils.domainnet_gaps import SEEDS, domainnet_runs, gap_by_seed, get_gap
from utils.logging_setup import setup_logging
from utils.paths import REPO_ROOT
from utils.results_table import RUNS_CSV, read_runs, sample_sd
from utils.style_distance import NEAR, read_style_distances

FIGURES_DIR = REPO_ROOT / "results" / "figures"
FIGURE_GAPS = ("G1", "G2")
PNG_DPI = 200
# Black and gray only, so the figure reads the same in grayscale.
INK = "black"
MUTED_INK = "0.45"
# One marker shape per style, in FD order. Styles 0.01 FD apart are too close for text labels
# beside the points, so a legend names them instead.
MARKERS = ("o", "s", "^", "D", "v", "P")


def plot_gap_vs_fd(runs, distances, gap):
    fd = [distance.fd_mean for distance in distances]
    fd_sd = [distance.fd_sd for distance in distances]
    values = [gap_by_seed(runs, gap, distance.style) for distance in distances]
    gap_mean = [mean(by_seed.values()) for by_seed in values]
    gap_sd = [sample_sd(list(by_seed.values())) for by_seed in values]

    figure, axes = plt.subplots(figsize=(6.0, 4.2))
    axes.axhline(0, color=MUTED_INK, linewidth=0.8, linestyle="--")
    for distance, marker, x, x_sd, y, y_sd in zip(distances, MARKERS, fd, fd_sd, gap_mean, gap_sd):
        # Filled markers are near styles and open markers far styles.
        face = INK if distance.group == NEAR else "white"
        axes.errorbar(
            x, y, xerr=x_sd, yerr=y_sd, fmt=marker, color=INK, markerfacecolor=face, markersize=7,
            elinewidth=1, capsize=3, label="{} ({}, FD {:.3f})".format(distance.style, distance.group, x),
        )
    axes.legend(fontsize=8, frameon=False, loc="best")

    axes.set_xlabel("FD to ImageNet, CLIP ViT-B/16 features (squared distance, no unit)")
    axes.set_ylabel("HM gap (percentage points)")
    axes.set_title("{}: {} on DomainNet styles".format(gap.name, gap.definition), fontsize=11)
    axes.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    figure.text(
        0.01, 0.01,
        "Points: mean over {0} training seeds; filled = near, open = far. Vertical bars: sample SD of the "
        "gap over the {0} seeds. Horizontal bars: sample SD of FD over the image samples.".format(len(SEEDS)),
        fontsize=7, color=MUTED_INK, wrap=True,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    return figure


def main():
    setup_logging()
    runs = domainnet_runs(read_runs(RUNS_CSV))
    distances = read_style_distances()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for name in FIGURE_GAPS:
        figure = plot_gap_vs_fd(runs, distances, get_gap(name))
        for suffix in ("pdf", "png"):
            path = FIGURES_DIR / "domainnet_gap_vs_fd_{}.{}".format(name, suffix)
            # No creation date in the PDF, so regenerating an unchanged figure gives an identical file.
            metadata = {"CreationDate": None} if suffix == "pdf" else None
            figure.savefig(path, dpi=PNG_DPI, metadata=metadata)
            logging.info("Wrote %s", path)
        plt.close(figure)


if __name__ == "__main__":
    main()

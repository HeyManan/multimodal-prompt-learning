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
from utils.style_distance import read_style_distances

FIGURES_DIR = REPO_ROOT / "results" / "figures"
FIGURE_GAPS = ("G1", "G2")
PNG_DPI = 200
# Black and gray only, so the figure reads the same in grayscale.
INK = "black"
MUTED_INK = "0.45"


def plot_gap_vs_fd(runs, distances, gap):
    fd = [distance.fd_mean for distance in distances]
    fd_sd = [distance.fd_sd for distance in distances]
    values = [gap_by_seed(runs, gap, distance.style) for distance in distances]
    gap_mean = [mean(by_seed.values()) for by_seed in values]
    gap_sd = [sample_sd(list(by_seed.values())) for by_seed in values]

    figure, axes = plt.subplots(figsize=(6.0, 4.2))
    axes.axhline(0, color=MUTED_INK, linewidth=0.8, linestyle="--")
    axes.errorbar(
        fd, gap_mean, xerr=fd_sd, yerr=gap_sd, fmt="o", color=INK, markersize=6,
        elinewidth=1, capsize=3,
    )
    for index, (distance, x, y) in enumerate(zip(distances, fd, gap_mean)):
        # Neighbors in FD order get labels on opposite sides, so close points do not overlap.
        offset = (6, 6) if index % 2 == 0 else (6, -14)
        axes.annotate(
            "{} ({})".format(distance.style, distance.group), (x, y), textcoords="offset points",
            xytext=offset, fontsize=9, color=INK,
        )
    axes.margins(x=0.15)

    axes.set_xlabel("FD to ImageNet, CLIP ViT-B/16 features (squared distance, no unit)")
    axes.set_ylabel("HM gap (percentage points)")
    axes.set_title("{}: {} on DomainNet styles".format(gap.name, gap.definition), fontsize=11)
    axes.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    figure.text(
        0.01, 0.01,
        "Points: mean over {0} training seeds. Vertical bars: sample SD of the gap over the {0} seeds. "
        "Horizontal bars: sample SD of FD over the image samples.".format(len(SEEDS)),
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
            figure.savefig(path, dpi=PNG_DPI)
            logging.info("Wrote %s", path)
        plt.close(figure)


if __name__ == "__main__":
    main()

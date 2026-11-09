"""DomainNet runs from results/runs.csv and the paired HM gaps between methods."""
import logging
from typing import NamedTuple

from utils.domainnet import STYLES
from utils.methods import DEFAULT_SEEDS, METHODS, NO_SEED
from utils.results_table import check_same_test_images, check_unique_runs
from utils.run_record import harmonic_mean

STYLE_DATASETS = {"domainnet_b2n_{}".format(style): style for style in STYLES}
# runs.csv stores seeds as text.
SEEDS = tuple(str(seed) for seed in DEFAULT_SEEDS)
HM_TOLERANCE = 1e-9
IDENTITY_TOLERANCE = 1e-9


class Gap(NamedTuple):
    name: str
    minuend: str
    subtrahend: str

    @property
    def definition(self):
        return "{} − {}".format(self.minuend, self.subtrahend)


GAPS = (
    Gap("G1", "MaPLe", "IVL-12"),
    Gap("G1'", "MaPLe-12", "IVL-12"),
    Gap("G1''", "MaPLe", "IVL-9"),
    Gap("G2", "MaPLe", "CoCoOp"),
    Gap("G3", "IVL-12", "CoCoOp"),
    Gap("G4", "MaPLe", "MaPLe-12"),
    Gap("G5", "IVL-12", "DLP-12"),
    Gap("G6", "DLP-12", "CoCoOp"),
)


def get_gap(name):
    gaps = {gap.name: gap for gap in GAPS}
    if name not in gaps:
        raise ValueError("Expected a gap in {}, found '{}'.".format(sorted(gaps), name))
    return gaps[name]


def method_seeds(method_name):
    return SEEDS if METHODS[method_name].is_trained else (NO_SEED,)


def run_hm(run):
    """HM of the run, after checking it against the hm column of runs.csv."""
    hm = harmonic_mean(float(run["base_acc"]), float(run["new_acc"]))
    if abs(hm - float(run["hm"])) > HM_TOLERANCE:
        raise ValueError(
            "Expected hm {} for {} {} seed {}, found {} in runs.csv.".format(
                hm, run["method"], run["dataset"], run["seed"], run["hm"]
            )
        )
    return hm


def domainnet_runs(runs):
    """Return {(method, style, seed): run} after checking that every expected run is there once."""
    selected = [run for run in runs if run["dataset"] in STYLE_DATASETS]
    check_unique_runs(selected)
    check_same_test_images(selected)
    by_key = {(run["method"], STYLE_DATASETS[run["dataset"]], run["seed"]): run for run in selected}

    expected = {(method, style, seed) for method in METHODS for style in STYLES for seed in method_seeds(method)}
    if set(by_key) != expected:
        raise RuntimeError(
            "Expected {} DomainNet runs, found {}. Missing: {}. Unexpected: {}.".format(
                len(expected), len(by_key), sorted(expected - set(by_key)), sorted(set(by_key) - expected)
            )
        )
    for run in by_key.values():
        run_hm(run)
    return by_key


def gap_by_seed(runs, gap, style):
    """{seed: HM of the minuend minus HM of the subtrahend, both from that seed}."""
    return {
        seed: run_hm(runs[(gap.minuend, style, seed)]) - run_hm(runs[(gap.subtrahend, style, seed)])
        for seed in SEEDS
    }


def check_gap_identity(runs):
    """G2 = G6 + G5 + G1: the steps CoCoOp to DLP-12 to IVL-12 to MaPLe must add up for every run."""
    largest = 0.0
    for style in STYLES:
        values = {name: gap_by_seed(runs, get_gap(name), style) for name in ("G1", "G2", "G5", "G6")}
        for seed in SEEDS:
            total = values["G6"][seed] + values["G5"][seed] + values["G1"][seed]
            difference = abs(values["G2"][seed] - total)
            if difference > IDENTITY_TOLERANCE:
                raise RuntimeError(
                    "Expected G2 = G6 + G5 + G1 for {} seed {}, found G2 {} and G6 + G5 + G1 {}.".format(
                        style, seed, values["G2"][seed], total
                    )
                )
            largest = max(largest, difference)
    logging.info(
        "G2 = G6 + G5 + G1 holds for %d styles x %d seeds; largest difference %.1e",
        len(STYLES), len(SEEDS), largest,
    )

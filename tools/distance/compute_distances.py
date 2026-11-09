"""Score each DomainNet style by its distance to ImageNet, from the cached CLIP features. CPU only.

For each sample seed, a style's features are compared with the ImageNet features of the same seed.
Writes results/distance/fd.csv and near_far.json; with --smoke, only $OUTPUT/smoke/distance/fd.csv.
Run tools/distance/extract_features.py first:

    PYTHONPATH="${PWD}" python tools/distance/compute_distances.py [--smoke]
"""
import argparse
import csv
import json
import logging
from statistics import mean

import numpy as np

from utils.distance_measures import frechet_distance, mean_and_covariance, nearest_neighbor_distance
from utils.logging_setup import setup_logging
from utils.paths import require_env_dir
from utils.results_table import sample_sd
from utils.run_record import git_commit
from utils.style_distance import (
    DISTANCE_RESULTS_DIR,
    FAR,
    FD_CSV,
    IMAGENET_SOURCE,
    NEAR,
    NEAR_FAR_JSON,
    NEAR_FAR_SIZE,
    SAMPLE_SEEDS,
    distance_run,
    feature_path,
    feature_record_path,
)

UNIT_NORM_TOLERANCE = 1e-9
RULE = "The {0} styles with the lowest mean FD to ImageNet are near, the {0} with the highest are far.".format(
    NEAR_FAR_SIZE
)


def load_features(run, source, seed):
    path = feature_path(run, source, seed)
    if not path.is_file():
        raise FileNotFoundError(
            "Expected features at {}, found none. Run tools/distance/extract_features.py first.".format(path)
        )
    features = np.load(path)["features"]
    largest_error = np.max(np.abs(np.linalg.norm(features, axis=1) - 1))
    if largest_error > UNIT_NORM_TOLERANCE:
        raise ValueError(
            "Expected unit-norm features in {}, found a norm that is off by {}.".format(path, largest_error)
        )
    return features


def load_record(run, source, seed):
    return json.loads(feature_record_path(run, source, seed).read_text())


def score_style(run, style, imagenet):
    """Return ({seed: FD}, {seed: NN distance}) for one style."""
    fd_by_seed, nn_by_seed = {}, {}
    for seed in run.sample_seeds:
        features = load_features(run, style, seed)
        reference = imagenet[seed]
        fd_by_seed[seed] = frechet_distance(*mean_and_covariance(features), *mean_and_covariance(reference))
        nn_by_seed[seed] = nearest_neighbor_distance(features, reference)
        logging.info("%s seed %d: FD %.6f, NN %.6f", style, seed, fd_by_seed[seed], nn_by_seed[seed])
    return fd_by_seed, nn_by_seed


def style_row(run, style, fd_by_seed, nn_by_seed):
    row = {"style": style}
    for measure, by_seed in (("fd", fd_by_seed), ("nn", nn_by_seed)):
        for seed in run.sample_seeds:
            row["{}_seed{}".format(measure, seed)] = by_seed[seed]
        row[measure + "_mean"] = mean(by_seed.values())
        row[measure + "_sd"] = sample_sd(list(by_seed.values()))
    row["images_per_style"] = run.images_per_style
    row["n_samples"] = len(run.sample_seeds)
    return row


def styles_by(rows, measure):
    return [row["style"] for row in sorted(rows, key=lambda row: row[measure + "_mean"])]


def split_near_far(styles_in_order):
    if len(styles_in_order) != 2 * NEAR_FAR_SIZE:
        raise ValueError("Expected {} styles to split, found {}.".format(2 * NEAR_FAR_SIZE, styles_in_order))
    return styles_in_order[:NEAR_FAR_SIZE], styles_in_order[NEAR_FAR_SIZE:]


def check_no_tie_at_boundary(rows):
    values = sorted(row["fd_mean"] for row in rows)
    if values[NEAR_FAR_SIZE - 1] == values[NEAR_FAR_SIZE]:
        raise RuntimeError(
            "Expected distinct mean FDs at the near/far boundary, found a tie at {}.".format(values[NEAR_FAR_SIZE])
        )


def add_groups_and_ranks(rows):
    """Set group from the FD rule, and the FD and NN ranks (1 = closest to ImageNet)."""
    by_fd, by_nn = styles_by(rows, "fd"), styles_by(rows, "nn")
    near, _ = split_near_far(by_fd)
    for row in rows:
        row["group"] = NEAR if row["style"] in near else FAR
        row["fd_rank"] = by_fd.index(row["style"]) + 1
        row["nn_rank"] = by_nn.index(row["style"]) + 1


def near_far_summary(run, rows):
    by_fd, by_nn = styles_by(rows, "fd"), styles_by(rows, "nn")
    near, far = split_near_far(by_fd)
    nn_near, nn_far = split_near_far(by_nn)
    sources = [IMAGENET_SOURCE] + list(run.styles)
    records = [load_record(run, source, seed) for source in sources for seed in run.sample_seeds]
    return {
        "rule": RULE,
        NEAR: near,
        FAR: far,
        "styles_by_fd": by_fd,
        # Reported only, to show whether the split depends on the measure. It never sets the groups.
        "nn_robustness": {
            "styles_by_nn": by_nn,
            NEAR: nn_near,
            FAR: nn_far,
            "same_split_as_fd": set(nn_near) == set(near),
        },
        "sample_seeds": list(run.sample_seeds),
        "images_per_style": run.images_per_style,
        "imagenet_images_per_class": run.imagenet_images_per_class,
        "backbone": sorted({record["backbone"] for record in records}),
        "feature_git_commits": sorted({record["git_commit"] for record in records}),
        "feature_gpus": sorted({record["gpu_name"] for record in records}),
        "distance_git_commit": git_commit(),
    }


def csv_fields(run):
    fields = ["style", "group", "fd_rank"]
    fields += ["fd_seed{}".format(seed) for seed in run.sample_seeds] + ["fd_mean", "fd_sd", "nn_rank"]
    fields += ["nn_seed{}".format(seed) for seed in run.sample_seeds] + ["nn_mean", "nn_sd"]
    return fields + ["images_per_style", "n_samples"]


def write_csv(run, rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        # The smoke run has no group or ranks, so those cells stay empty.
        writer = csv.DictWriter(file, fieldnames=csv_fields(run), restval="")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["fd_mean"]))
    logging.info("Wrote %s", path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="score the smoke run's one style")
    parser.add_argument("--sample-seeds", type=int, nargs="+", default=list(SAMPLE_SEEDS))
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()
    run = distance_run(require_env_dir("OUTPUT"), args.smoke, args.sample_seeds)
    logging.info("Styles %s, sample seeds %s", run.styles, run.sample_seeds)

    imagenet = {seed: load_features(run, IMAGENET_SOURCE, seed) for seed in run.sample_seeds}
    rows = [style_row(run, style, *score_style(run, style, imagenet)) for style in run.styles]

    if args.smoke:
        write_csv(run, rows, run.output_dir / "fd.csv")
        logging.info("Style distance smoke run finished.")
        return

    check_no_tie_at_boundary(rows)
    add_groups_and_ranks(rows)
    write_csv(run, rows, FD_CSV)
    DISTANCE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    NEAR_FAR_JSON.write_text(json.dumps(near_far_summary(run, rows), indent=2) + "\n")
    logging.info("Wrote %s", NEAR_FAR_JSON)


if __name__ == "__main__":
    main()

"""Settings and file locations of the style distance to ImageNet, and readers of its results."""
import csv
import json
from pathlib import Path
from typing import NamedTuple, Tuple

from utils.domainnet import STYLES
from utils.paths import REPO_ROOT, SMOKE_FOLDER

IMAGENET_SOURCE = "imagenet"
IMAGENET_NUM_CLASSES = 1000
IMAGES_PER_STYLE = 5000
IMAGENET_IMAGES_PER_CLASS = 5
SAMPLE_SEEDS = (1, 2, 3)
NEAR_FAR_SIZE = 3

SMOKE_STYLE = "real"
SMOKE_SAMPLE_SEED = 1
# More images than the 512 feature dimensions, so both covariances have full rank.
SMOKE_IMAGES_PER_STYLE = 1000
SMOKE_IMAGENET_IMAGES_PER_CLASS = 1

# Relative to $DATA. Files from the Hugging Face ILSVRC/imagenet-1k dataset; see README.md.
IMAGENET_PARQUET_DIR = Path("imagenet_hf") / "data"
IMAGENET_PARQUET_GLOB = "validation-*.parquet"

# The test loader settings of every trainer config.
FEATURE_BATCH_SIZE = 100
NUM_WORKERS = 8

DISTANCE_FOLDER = "distance"
DISTANCE_RESULTS_DIR = REPO_ROOT / "results" / "distance"
FD_CSV = DISTANCE_RESULTS_DIR / "fd.csv"
NEAR_FAR_JSON = DISTANCE_RESULTS_DIR / "near_far.json"
NEAR, FAR = "near", "far"


class DistanceRun(NamedTuple):
    styles: Tuple[str, ...]
    sample_seeds: Tuple[int, ...]
    images_per_style: int
    imagenet_images_per_class: int
    output_dir: Path  # cached features, and the smoke run's results


def distance_run(output_root, smoke, sample_seeds):
    if smoke:
        return DistanceRun(
            styles=(SMOKE_STYLE,),
            sample_seeds=(SMOKE_SAMPLE_SEED,),
            images_per_style=SMOKE_IMAGES_PER_STYLE,
            imagenet_images_per_class=SMOKE_IMAGENET_IMAGES_PER_CLASS,
            output_dir=output_root / SMOKE_FOLDER / DISTANCE_FOLDER,
        )
    return DistanceRun(
        styles=STYLES,
        sample_seeds=tuple(sample_seeds),
        images_per_style=IMAGES_PER_STYLE,
        imagenet_images_per_class=IMAGENET_IMAGES_PER_CLASS,
        output_dir=output_root / DISTANCE_FOLDER,
    )


def feature_path(run, source, seed):
    return run.output_dir / "features" / "{}_seed{}.npz".format(source, seed)


def feature_record_path(run, source, seed):
    return feature_path(run, source, seed).with_suffix(".json")


class StyleDistance(NamedTuple):
    style: str
    group: str
    fd_mean: float
    fd_sd: float


def read_style_distances():
    """Return the 6 styles in order of mean FD, after checking fd.csv against near_far.json."""
    for path in (FD_CSV, NEAR_FAR_JSON):
        if not path.is_file():
            raise FileNotFoundError(
                "Expected {}, found none. Run slurm/style_distance.sbatch first.".format(path)
            )
    with FD_CSV.open(newline="") as file:
        rows = list(csv.DictReader(file))
    distances = sorted(
        (StyleDistance(row["style"], row["group"], float(row["fd_mean"]), float(row["fd_sd"])) for row in rows),
        key=lambda distance: distance.fd_mean,
    )
    found_styles = [distance.style for distance in distances]
    if sorted(found_styles) != sorted(STYLES):
        raise ValueError("Expected one row per style {} in {}, found {}.".format(STYLES, FD_CSV, found_styles))

    near_far = json.loads(NEAR_FAR_JSON.read_text())
    for distance in distances:
        expected = NEAR if distance.style in near_far[NEAR] else FAR
        if distance.group != expected:
            raise ValueError(
                "Expected group '{}' for {} as in {}, found '{}' in {}.".format(
                    expected, distance.style, NEAR_FAR_JSON, distance.group, FD_CSV
                )
            )
    return distances
